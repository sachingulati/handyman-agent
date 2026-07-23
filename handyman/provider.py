"""Which model runs a job, and where.

Both are the caller's decision. Only the caller knows whether a task
wants the strongest available model or the cheapest, whether it may leave
the machine, and whether latency or privacy matters more. Guessing on
their behalf produces a tool that is convenient right up until it makes
the wrong call silently.

So `provider` and `model` are arguments. What is left here is validation:
refusing a request that cannot work, and saying why in terms of what to
do about it.

One default is deliberately conservative. When the caller says nothing,
work stays local - a busy queue waits rather than being redirected off
the machine. Hosted work carries the task text and whatever files the
model reads to a third party, and that is not something to arrange for
someone who did not ask.
"""

from handyman import config

LOCAL = "local"
HOSTED = "hosted"
PROVIDERS = (LOCAL, HOSTED)


class ProviderUnavailable(Exception):
    """A job cannot run as requested, with an explanation of what to do."""


def _hosted_ready(cfg) -> bool:
    return bool(cfg.api_key_env) and bool(config.api_key_for(cfg))


def validate(cfg, resolved, local_available: bool) -> None:
    """Refuse a resolved model that cannot actually be reached.

    The registry has already decided where the request goes; all that is
    left is to fail early, with a message about what to do, rather than
    letting the job start and die on a connection error.
    """
    if resolved.provider.hosted:
        if not resolved.provider.api_key():
            raise ProviderUnavailable(
                f"{resolved.name} runs on {resolved.provider.name}, which needs "
                f"an API key - set {resolved.provider.api_key_env} and try again"
            )
        return
    if not local_available:
        raise ProviderUnavailable(
            f"{resolved.name} runs on the local model server, which is not "
            "reachable. Start it, or ask for a hosted model - `handyman models` "
            "lists what is available."
        )


def choose(cfg, requested: str | None, local_available: bool,
           at_capacity: bool) -> str:
    """Validate the caller's choice of provider, or apply the safe default.

    requested       - "local", "hosted", or None to let the default apply
    local_available - whether the local model server answered
    at_capacity     - whether the local concurrency cap is already reached
    """
    local_configured = bool(getattr(cfg, "tiers", None))

    if requested is not None:
        if requested not in PROVIDERS:
            raise ProviderUnavailable(
                f"unknown provider {requested!r} - expected one of {', '.join(PROVIDERS)}"
            )
        if requested == HOSTED:
            if not _hosted_ready(cfg):
                raise ProviderUnavailable(
                    "hosted was requested but there is no API key - set "
                    f"{cfg.api_key_env or 'an api_key_env in the config'} "
                    "and try again"
                )
            return HOSTED
        if not local_available:
            raise ProviderUnavailable(
                "local was requested but the model server is not reachable - "
                "start it, or request the hosted provider instead"
            )
        # A busy local queue is not a failure: the job waits its turn.
        return LOCAL

    # Nothing requested. Prefer local, and never silently leave the machine.
    if local_configured and local_available:
        return LOCAL
    if not local_configured and _hosted_ready(cfg):
        # Hosted-only install: there is nothing to opt out of.
        return HOSTED
    if _hosted_ready(cfg):
        raise ProviderUnavailable(
            'the local model server is not reachable. Start it, or pass '
            'provider="hosted" to run this task on the hosted model - note '
            "that doing so sends the task, and any files it reads, off this "
            "machine."
        )
    raise ProviderUnavailable(
        "no local model server and no hosted API key - run `handyman setup`"
    )


def resolve_model(cfg, requested: str | None) -> str:
    """The model this job should use.

    A requested model is taken as given, even when this install has never
    seen it: the caller may know about one that was pulled since the
    config was written, and refusing it would make the argument useless.
    """
    if requested:
        return requested
    tiers = getattr(cfg, "tiers", None)
    if not tiers:
        raise ProviderUnavailable(
            "no model was requested and no model is configured - pass a model, "
            "or run `handyman setup`"
        )
    return tiers[0].model
