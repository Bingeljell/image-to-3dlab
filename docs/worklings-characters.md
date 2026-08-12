# Characters: Families & Classes

This is the single home for the **identity layer** of Worklings: creature families, the species within them, the class roster, and — once they exist — skills and abilities. When a new species or class is added, it lands here first.

The **mechanics** behind these identities (XP math, stat-growth weights, caps, persistence) live in [Progression](progression.md); this doc records who the characters *are*, that one records how the numbers *move*. The two link to each other rather than repeating each other.

## Families

A family is the species axis — what MMOs would call a race. Family now carries a **light mechanical identity** — a stat lean and a passive (see [Family–class affinity](#familyclass-affinity)) — on top of its art, silhouette, and world flavour.

This is a deliberate **revision** of the original *purely cosmetic* families. That decision was made before combat existed and to protect a "switching family preserves everything" promise. Now that dungeons ship, race has earned the right to *mean* something in a fight. The touch is kept deliberately **soft** — a lean and a passive, never a hard class lock — so switching family still preserves name, needs, favourites, relationship, level, XP, and class; only the small racial lean and passive move with it. (See [Progression](progression.md#families-carry-a-soft-mechanical-identity) for the mechanical side.)

Every species ships the same twelve-frame pose contract, so all moods, reactions, and transitions read identically whichever Workling is active.

The five families form five thematic lanes — nature, elements, machinery, energy, cosmos:

| Family | Lane | World flavor | Current species |
| --- | --- | --- | --- |
| **Wildkin** | Nature | Creatures shaped by living ecosystems and natural magic. | Moss-fox |
| **Elemental** | Elements | Creatures whose elemental affinity is part of their anatomy. | Ember-newt |
| **Relicborn** | Machinery | Creatures bonded to ancient mechanisms, relics, or rune-powered artifacts. | Keyback pangolin |
| **Glitchkin** | Energy | Creatures born in the seams between systems — signal, speed, and unstable reality. | — *(coming soon — art pending)* |
| **Bloomglass** | Cosmos | Creatures of celestial stillness — starlight, crystal, and refracted moonlight. | — *(coming soon — art pending)* |

All five families now exist in code (`PetFamily`). Glitchkin and Bloomglass are complete
except for their **art**, so `PetFamily.hasArt` marks them and the family menu lists them
**"(coming soon)" and greyed out** — visible, so the roster reads as five lanes, but not
selectable until a sheet exists. Both the menu state and the selection handler key off
`hasArt`, so each un-greys on its own the moment its art lands. Their stat lean and passive remain
designed-not-built, the same as the other three families'. Their arrival is what unblocks
the two stats — Guard and Agility — whose [item attunement](items.md#family-attunement--the-soft-synergy-layer) had no
family to point at.

### Species

**Species is purely cosmetic** — the critters within a family share its lean, passive, and every class option; they change only the *look*. The full designed catalogue (five-plus per family) lives in the **[Race & Creature Roster](worklings_race_creature_roster.md)**; this section records only which are *implemented* (have art) vs designed.

Implemented today (one per shipped family):

- **Moss Fox** (Wildkin) — a fox shaped by living woodland magic, moss and growth woven into its coat.
- **Ember Newt** (Elemental) — a newt whose inner fire is part of its body, not an effect on it.
- **Key-back Pangolin** (Relicborn) — a pangolin whose scales are bonded to an ancient rune-powered relic, a key grown into its back.

Every other critter in the roster (e.g. Wildkin's Canopy Elephant, Glitchkin's Sparktail, Bloomglass's Starpetal Fawn) is **designed, not yet built**. Adding one means: pick its family, deliver the twelve-frame pose contract, and it's selectable. Species selection today is Pixel switching appearance; a full adoption flow remains a later slice.

> Because species is cosmetic, a hulking critter can be a glass-cannon and a tiny one a tank — look and role are decoupled. A later UI cue (class shown beside species) should make that freedom read as intentional.

### The two newest families (design-stage)

Glitchkin and Bloomglass round the roster to five balanced lanes; both are designed but not yet built (no sprites).

- **Glitchkin** (also *Signalborn*) — beings shaped by electricity, waves, speed, and interference; born where signals overlap and portals misalign. Sleek, fast, asymmetric silhouettes broken by glowing pulse-lines; darting, flickering movement with brief afterimages. Where Wildkin feel *alive* and Relicborn feel *constructed*, Glitchkin feel **transmitted**.
- **Bloomglass** (also *Astralglass* / *Astralward*) — beings of celestial stillness: starlight, crystal growth, refracted moonlight. Smooth, semi-translucent, luminous-from-within forms; slow, poised, deliberate motion. Read as **big floaty masses** — of soaking presence or of healing energy alike.

## Classes

Class is the **primary mechanical-identity** axis. Any species can still be any class — the family lean only *tilts* the fit (see [Family–class affinity](#familyclass-affinity)), it never forbids a class. Class decides which of the five stats is the signature stat — the one that grows fastest per level — while the other four keep growing at a slower, steady rate. Growth weights and the leveling math live in [Progression](progression.md#class).

Every class name is deliberately dual-coded: a term with real currency in modern work/maker culture that also carries its own mythic or abstract weight, independent of any RPG convention. The roster maps one class per stat, each filling a traditional RPG role:

| Stat | Class | Role | Flavor |
| --- | --- | --- | --- |
| Vitality | **Wellspring** | Healer / Support | The source others draw on — sustains, restores, never runs dry. |
| Power | **Juggernaut** | Heavy offense | Hits like an unstoppable force — raw, overwhelming offense. |
| Guard | **Aegis** | Tank | The shield everyone stands behind — mitigates, endures, protects. |
| Agility | **Maverick** | Finesse offense | Moves fast, breaks convention — quick, decisive, takes the opening first. |
| Wit | **Tinkerer** | Mage-equivalent | Technology so advanced it might as well be magic — clever, inventive, otherworldly effective. |

Class is freely reassignable for now, the same way family is — nothing yet (no ability trees, no gear) needs protecting across a swap. Once abilities lock to a class, reassignment may become a deliberate, costed action; that is a later revisit recorded in [Progression](progression.md#class).

## Family–class affinity

Family and class are **soft-coupled**: each family fits some classes better than others, but never *only* those. The model is deliberately light so it adds identity without removing freedom or making any class unreachable — a hard race→class lock was considered and rejected (a one-character game shouldn't foreclose builds forever, and every class must stay pickable).

**Only two things are locked per family:**

- a **Primary** class — its natural best fit, where its stat lean and passive line up; and
- a **Weak** class — its anti-fit, where they pull against it.

The **other three classes are open** — neutral, fully valid picks. The trick that avoids a per-combo bonus table: each family has **one signed stat lean** — a small **`+` on its primary class's stat and a `−` on its weak class's stat** — plus **one passive**. That single signed lean yields three genuine tiers from one knob: the **Primary** rides the `+`, the **Weak** takes the `−` (a real but *mild* penalty — a tilt, never a lock), and the **middle three** are untouched (neutral trade-offs). Magnitudes stay small (held knobs), so a "weak" combo is a couple of points behind, never unplayable — a Glitchkin *can* be a Juggernaut, it's just no Relicborn.

| Family | 🔒 Primary | 🔒 Weak | Open (player's pick) | Signed lean · passive |
| --- | --- | --- | --- | --- |
| **Wildkin** | Wellspring | Juggernaut | Aegis · Maverick · Tinkerer | +Vitality / −Power · **Regrowth** |
| **Elemental** | Tinkerer | Aegis | Juggernaut · Maverick · Wellspring | +Wit / −Guard · **Overload** |
| **Relicborn** | Juggernaut | Maverick | Aegis · Tinkerer · Wellspring | +Power / −Agility · **Relic Plating** |
| **Glitchkin** | Maverick | Juggernaut | Aegis · Tinkerer · Wellspring | +Agility / −Power · **Phase Flicker** |
| **Bloomglass** | Aegis | Maverick | Wellspring · Tinkerer · Juggernaut | +Guard / −Agility · **Refraction Ward** |

Every class is the Primary of exactly one family (so no class lacks a home), and every family's `−` lands on a *different* class's signature stat, so "weak" always means something concrete.

**Three flavours of damage** are what make this fit cleanly — "DPS" is not one class:

- **Juggernaut** — *physical* brute (Power). Relicborn's machine pistons.
- **Maverick** — *agile* strike (Agility). Glitchkin's spacetime zaps; a Glitchkin's evasive survivability (miss-based "tanking") lives here, in Agility, not in Aegis's raw absorption.
- **Tinkerer** — *magic* damage (Wit). Elemental's fireblast reads as a spell, which is why the Elemental — clever enough to bend the elements to its will — is the Mage.

**Bloomglass** flexes its Aegis primary two ways — a mass that *soaks* damage or a mass of *healing* energy (Wellspring is its strongest open option).

Lean sizes and the weak-class penalty are held knobs. The passives are specced in [Family passives](#family-passives) below.

**Mage note (v1):** "Mage" is a *class* thing — **Tinkerer** (Wit → magic damage) is the mage, and **Wellspring** also runs on Wit (for healing); any family can pick either. Among the *families*, only **Elemental** carries a magic-boosting passive in v1 (Overload), because elemental magic is its identity. A Wit-flavoured passive for Glitchkin/Wildkin (a "secondary caster" race) is deferred, not a restriction on who can cast.

## Skills & abilities

The v1 model is **locked** — full detail (currencies, skill tree, scaling, cost) lives in [Abilities](abilities.md). Two tracks, split by source:

- **Class → active abilities** — the "extra button." Level- and class-gated, unlocked/ranked with **Skill Points** (one of two level-granted pools; the other is **Stat Points** for manual stat allocation). Each class's first ability *replaces* the generic Signature. The five: Wellspring **Second Wind**, Juggernaut **Overbear**, Aegis **Bulwark**, Maverick **Flurry**, Tinkerer **Exploit**.
- **Family → passive traits** — automatic, no button (below).

Both ride the shared **status-effect** primitive (built) and the **trigger-hook** layer (designed) — see [Abilities](abilities.md).

### Family passives

One automatic passive per family (v1) — the racial identity in a fight. It's what lets a family cover a gap so the player can go all-in on their class (a Relicborn Juggernaut solo-levels safely on Relic Plating; a Wildkin Aegis tanks *and* self-heals). Magnitudes held.

| Family | Passive | Effect |
| --- | --- | --- |
| **Wildkin** | **Regrowth** | Small HP regen every round (nature sustain). |
| **Elemental** | **Overload** | Chance on hit to add a small elemental proc (bonus magic damage). |
| **Relicborn** | **Relic Plating** | Flat damage reduction — constructed durability. |
| **Glitchkin** | **Phase Flicker** | Extra evasion, plus an occasional full dodge after being hit. |
| **Bloomglass** | **Refraction Ward** | A small damage-absorbing shield that refreshes / reflects a portion. |

A passive *line* (multiple, chosen with Skill Points) and race *actives* are deferred past v1.
