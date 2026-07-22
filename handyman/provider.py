"""Deciding whether a job runs locally or on a hosted model.

The rule is deliberately conservative in one direction: a job never goes
to a hosted provider by accident. Sending work off the machine means the
task text, and whatever files the model reads, leave it. Someone who
pasted an API key once, months ago, should not discover that an unrelated
local outage has been quietly forwarding their code ever since.

So the caller opts in, and that opt-in covers both reasons hosted might
be reached for - the local queue being busy, and the local server being
down. One rule, not two.
"""

from handyman import config


class ProviderUnavailable(Exception):
    """No provider can run this job, with an explanation of what to do."""


def choose(cfg, local_available: bool, at_capacity: bool,
           allow_hosted: bool) -> str:
    """Return "local" or "hosted", or raise saying why neither works.

    local_available - whether the local model server answered
    at_capacity    - whether the local concurrency cap is already reached
    allow_hosted   - the caller's explicit consent to leave the machine
    """
    hosted_ready = bool(cfg.api_key_env) and bool(config.api_key_for(cfg))
    local_configured = bool(getattr(cfg, "tiers", None))

    # Nothing local to run: hosted is the only option, and there is
    # nothing for the caller to have opted out of.
    if not local_configured:
        if hosted_ready:
            return "hosted"
        raise ProviderUnavailable(
            "no model tiers are configured and no hosted API key is set - "
            "run `handyman setup` to choose a model"
        )

    if local_available and not at_capacity:
        return "local"

    if at_capacity and local_available:
        # Queueing is the established behaviour and stays the default.
        return "hosted" if (allow_hosted and hosted_ready) else "local"

    # Local is unreachable.
    if allow_hosted and hosted_ready:
        return "hosted"

    if hosted_ready:
        raise ProviderUnavailable(
            "the local model server is not reachable. Start it, or re-submit "
            "with allow_hosted=True to send this task to the hosted provider "
            "instead - note that doing so sends the task and any files it "
            "reads off this machine."
        )
    raise ProviderUnavailable(
        "the local model server is not reachable - start it, or configure a "
        "hosted provider with `handyman setup`"
    )
