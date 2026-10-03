// A Cancel button that works from the moment a job is asked for. The job id only comes
// back with the response, and an upload can take a while, so a click before then is kept
// and sent the moment the job exists. One per job: a click never reaches an older job.
export function earlyCancel(send) {
  let jobId = null;
  let wanted = false;
  return {
    started(id) {
      jobId = id;
      if (wanted) send(id);
    },
    // 'sent' when the job exists, 'queued' when it will be cancelled once it does.
    click() {
      if (jobId) {
        send(jobId);
        return 'sent';
      }
      wanted = true;
      return 'queued';
    },
  };
}
