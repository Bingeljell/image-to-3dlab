"""Small, deterministic checks of Hunyuan's CPU preprocessing and scheduling."""

import numpy as np
import pytest
from PIL import Image

from hunyuan_mlx.paint.hy3dpaint_mlx.scheduler import DDIMScheduler, UniPCScheduler
from hunyuan_mlx.shape.hy3dmlx.preprocess import load_image, recenter
from hunyuan_mlx.shape.hy3dmlx.sampler import consistency_sigmas, flow_match_sigmas


def test_rgb_is_centered_with_a_white_border():
    rgb = np.full((3, 5, 3), [100, 20, 0], dtype=np.uint8)
    result = recenter(rgb, border_ratio=0.2)
    assert result.shape == (5, 5, 3)
    np.testing.assert_array_equal(result[1:3, :4], np.broadcast_to(rgb[0, 0], (2, 4, 3)))
    assert (result[0] == 255).all()
    assert (result[:, 4] == 255).all()


def test_rgba_recenter_composites_alpha_against_white():
    rgba = np.zeros((5, 5, 4), dtype=np.uint8)
    rgba[1:4, 1:4] = [0, 0, 0, 128]
    result = recenter(rgba, border_ratio=0.4)
    np.testing.assert_array_equal(result[1:4, 1:4], np.full((3, 3, 3), 126, dtype=np.uint8))
    assert (result[[0, -1]] == 255).all()


def test_recenter_rejects_an_object_with_no_height():
    pixels = np.zeros((5, 5, 4), dtype=np.uint8)
    pixels[2, :, 3] = 255
    with pytest.raises(ValueError, match="input image is empty"):
        recenter(pixels)


def test_file_and_pillow_preprocessing_match(tmp_path):
    image = Image.new("RGB", (4, 4), (0, 128, 255))
    path = tmp_path / "colors.png"
    image.save(path)
    from_path = load_image(str(path), size=2, border_ratio=0)
    assert from_path.dtype == np.float32
    assert from_path.shape == (1, 3, 2, 2)
    np.testing.assert_array_equal(from_path, load_image(image, size=2, border_ratio=0))
    np.testing.assert_allclose(from_path[0, :, 0, 0], [-1, 1 / 255, 1], atol=1e-7)


def test_flow_schedule_shifts_the_interior_and_keeps_both_endpoints():
    sigma, with_terminal = flow_match_sigmas(3, shift=3)
    assert sigma.dtype == with_terminal.dtype == np.float32
    np.testing.assert_array_equal(sigma, [0, 0.75, 1])
    np.testing.assert_array_equal(with_terminal, [0, 0.75, 1, 1])


def test_distilled_schedule_uses_the_phased_subset():
    sigma, with_terminal = consistency_sigmas(3, pcm_timesteps=5, num_train_timesteps=10)
    np.testing.assert_allclose(sigma, np.array([0, 1, 5]) / 9)
    np.testing.assert_array_equal(with_terminal[:-1], sigma)
    assert with_terminal[-1] == 1


def test_ddim_recovers_clean_sample_via_interior_and_terminal_steps():
    scheduler = DDIMScheduler(num_train=12)
    scheduler.set_timesteps(3)
    np.testing.assert_array_equal(scheduler.timesteps, [11, 7, 3])
    assert scheduler.init_noise_sigma() == 1
    clean = np.array([0.25, -0.5])
    noise = np.array([0.75, 0.125])
    for t, previous in zip(scheduler.timesteps, [7, 3, -1], strict=True):
        a, b = np.sqrt(scheduler.acp[t]), np.sqrt(1 - scheduler.acp[t])
        sample = a * clean + b * noise
        velocity = a * noise - b * clean
        assert scheduler.scale_model_input(sample, t) is sample
        prev_alpha = scheduler.acp[previous] if previous >= 0 else 1
        expected = np.sqrt(prev_alpha) * clean + np.sqrt(1 - prev_alpha) * noise
        np.testing.assert_allclose(scheduler.step(velocity, t, sample), expected)


@pytest.mark.parametrize(("solver_type", "order"), [("bh1", 2), ("bh2", 3)])
def test_unipc_transports_a_constant_clean_prediction_and_resets(solver_type, order):
    scheduler = UniPCScheduler(num_train=20, solver_type=solver_type, solver_order=order)
    scheduler.set_timesteps(5)
    np.testing.assert_array_equal(scheduler.timesteps, [19, 15, 11, 7, 3])
    clean, noise = 0.2, -0.7
    sigmas = scheduler.sigmas.astype(float)
    sample = (clean + sigmas[0] * noise) / np.hypot(1, sigmas[0])
    assert scheduler.init_noise_sigma() == 1
    assert scheduler.scale_model_input(sample, 19) is sample
    orders = []
    # bh1 uses an infinite coefficient at the zero-sigma endpoint.
    timesteps = scheduler.timesteps if solver_type == "bh2" else scheduler.timesteps[:-1]
    for index, timestep in enumerate(timesteps):
        sigma = sigmas[index]
        v = (noise - sigma * clean) / np.hypot(1, sigma)
        sample = scheduler.step(v, timestep, sample)
        next_sigma = sigmas[index + 1]
        assert sample == pytest.approx((clean + next_sigma * noise) / np.hypot(1, next_sigma))
        orders.append(scheduler.this_order)
    assert orders == ([1, 2, 2, 2] if order == 2 else [1, 2, 3, 2, 1])
    assert scheduler.lower_order_nums == order
    scheduler.set_timesteps(2)
    assert scheduler.last_sample is None
    assert scheduler.model_outputs == [None] * order
    assert scheduler.lower_order_nums == 0


def test_unipc_uses_last_step_when_timestep_is_absent():
    scheduler = UniPCScheduler(num_train=12)
    scheduler.set_timesteps(3)
    assert scheduler.step(0.0, -1, 1.0) == pytest.approx(1 / np.hypot(1, scheduler.sigmas[-2]))
    assert scheduler.timestep_list[-1] == -1


def test_unipc_starts_at_second_duplicate_timestep():
    scheduler = UniPCScheduler(num_train=3, disable_corrector=[1], lower_order_final=False)
    scheduler.set_timesteps(7)
    assert list(scheduler.timesteps[:2]) == [2, 2]
    assert np.isfinite(scheduler.step(0.0, 2, 1.0))
    assert scheduler._step_index == 2
    assert scheduler.this_order == 1


def test_unipc_rejects_unsupported_solver():
    scheduler = UniPCScheduler(num_train=12, solver_type="unknown")
    scheduler.set_timesteps(3)
    with pytest.raises(NotImplementedError, match="unknown"):
        scheduler.step(0.0, 11, 1.0)


@pytest.mark.parametrize("disable_corrector", [[], [0, 1]])
def test_unipc_noise_prediction_transports_interior_samples(disable_corrector):
    scheduler = UniPCScheduler(num_train=20, predict_x0=False, disable_corrector=disable_corrector)
    scheduler.set_timesteps(5)
    clean, noise = -0.4, 0.8
    sigmas = scheduler.sigmas.astype(float)
    sample = (clean + sigmas[0] * noise) / np.hypot(1, sigmas[0])
    for index, timestep in enumerate(scheduler.timesteps[:-1]):
        sigma = sigmas[index]
        v = (noise - sigma * clean) / np.hypot(1, sigma)
        sample = scheduler.step(v, timestep, sample)
        next_sigma = sigmas[index + 1]
        assert sample == pytest.approx((clean + next_sigma * noise) / np.hypot(1, next_sigma))


def test_euler_updates_accumulate_and_skip_zero_width_steps():
    from hunyuan_mlx.shape.hy3dmlx.sampler import euler_steps

    state = np.array([2.0, 3.0])
    visits = []
    sigmas = (np.array([0.0, 0.25, 0.25, 0.75]), np.array([0.0, 0.25, 0.25, 0.75, 1.0]))

    def velocity(sample, timestep):
        visits.append(("velocity", timestep))
        return sample * (1 + timestep)

    def materialize(sample):
        visits.append(("update", sample[0]))

    def progress(step, total):
        visits.append(("progress", step, total))

    output = euler_steps(state, velocity, sigmas, materialize=materialize, progress=progress)
    np.testing.assert_array_equal(output, state * 1.25 * 1.625 * 1.4375)
    assert visits == [
        ("velocity", 0.0), ("update", 2.5), ("progress", 1, 4),
        ("velocity", 0.25), ("update", 4.0625), ("progress", 3, 4),
        ("velocity", 0.75), ("update", 5.83984375), ("progress", 4, 4),
    ]


def test_euler_works_without_progress_and_leaves_input_unchanged():
    from hunyuan_mlx.shape.hy3dmlx.sampler import euler_steps

    state = np.array([4.0])
    result = euler_steps(state, lambda sample, _t: -sample, flow_match_sigmas(2),
                         materialize=lambda _sample: None)
    np.testing.assert_array_equal(result, [0])
    np.testing.assert_array_equal(state, [4])


def test_dino_normalization_returns_channels_last_imagenet_values():
    from hunyuan_mlx.shape.hy3dmlx.preprocess import dino_pixels

    img = np.broadcast_to(np.array([-1, 0, 1], dtype=np.float32)[:, None, None], (3, 1, 1))
    tensor = dino_pixels(img, image_size=2)
    assert tensor.shape == (1, 2, 2, 3)
    assert tensor.dtype == np.float32
    normalized = (np.array([0, 0.5, 1]) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np.testing.assert_allclose(tensor[0], np.broadcast_to(normalized, (2, 2, 3)), atol=1e-7)
