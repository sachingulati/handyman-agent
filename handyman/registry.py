"""Which models exist, and where each one lives.

A model name on its own is not enough to send a request. `gemma-4-31b-it`
exists only on a hosted endpoint; `huihui_ai/gemma-4-abliterated:26b`
exists only on the local server. Treating the model and its location as
separate settings lets a caller ask for a combination that cannot work,
and the failure arrives late and confusingly - as a 404 from whichever
endpoint happened to be configured.

So a model is registered *with* its provider, and asking for a model
selects the endpoint too.

Most models never need registering by hand. Both kinds of provider can
list what they hold - the local server through its tag list, a hosted one
through the OpenAI-compatible models endpoint - so anything already
installed or available is addressable by its own name. Explicit entries
exist for giving something a short alias, or for pinning a specific model
when several would otherwise match.
"""

from dataclasses import dataclass

import requests

LOCAL_LIST_PATH = "/api/tags"
HOSTED_LIST_PATH = "/models"
DISCOVERY_TIMEOUT = 15


class ModelUnavailable(Exception):
    """A model was asked for that no configured provider offers."""


@dataclass(frozen=True)
class Provider:
    name: str
    host: str
    chat_path: str = "/v1/chat/completions"
    api_key_env: str = ""
    # Off by default. A hosted catalogue mixes free models with ones that
    # bill per token at wildly different rates, and a delegating model
    # picking from that list has no way to know which is which. Turn it on
    # for a flat-rate or self-hosted endpoint where every model is equally
    # safe to reach for.
    allow_discovered: bool = False

    @property
    def hosted(self) -> bool:
        """A provider needing a key is remote; one without is the local server."""
        return bool(self.api_key_env)

    def api_key(self):
        import os

        return os.environ.get(self.api_key_env) or None if self.api_key_env else None


@dataclass(frozen=True)
class Model:
    """A model a caller can ask for, and where the request goes.

    `cost` is advisory and defaults to "unknown" rather than "free":
    silence about price must never read as a promise that there is none.

    `enabled` is the enforced part, and describes a decision rather than a
    property of the model: a provider may offer something perfectly
    capable that nobody has agreed to pay for. Such a model stays visible,
    so it can be found and chosen, but jobs cannot reach it.
    """
    name: str
    provider: Provider
    model_id: str
    cost: str = "unknown"
    enabled: bool = True
    note: str = ""


def providers_from_config(cfg) -> list[Provider]:
    """Providers declared in config, plus the implicit one from the flat
    settings that predate this - so an existing config keeps working."""
    declared = getattr(cfg, "providers", None) or {}
    providers = [
        Provider(
            name=name,
            host=spec["host"],
            chat_path=spec.get("chat_path", "/v1/chat/completions"),
            api_key_env=spec.get("api_key_env", ""),
            allow_discovered=bool(spec.get("allow_discovered", False)),
        )
        for name, spec in declared.items()
    ]
    if not providers:
        providers.append(Provider(
            name="hosted" if cfg.api_key_env else "local",
            host=cfg.ollama_host,
            chat_path=cfg.chat_path,
            api_key_env=cfg.api_key_env,
        ))
    return providers


def discover(provider: Provider, timeout: int = DISCOVERY_TIMEOUT) -> list[str]:
    """Model ids a provider currently offers.

    Returns an empty list rather than raising: discovery is a convenience,
    and an unreachable provider should not stop a caller naming a model
    explicitly.
    """
    headers = {"Authorization": f"Bearer {provider.api_key()}"} if provider.hosted else None
    path = HOSTED_LIST_PATH if provider.hosted else LOCAL_LIST_PATH
    try:
        resp = requests.get(f"{provider.host}{path}", headers=headers, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return []

    if provider.hosted:
        # OpenAI-compatible listings use {"data": [{"id": ...}]}, and some
        # return a namespaced id such as "models/gemma-4-31b-it".
        return [str(m.get("id", "")).rsplit("/", 1)[-1]
                for m in payload.get("data", []) if m.get("id")]
    return [m["name"] for m in payload.get("models", []) if m.get("name")]


def registered(cfg) -> list[Model]:
    """Explicit entries from the config's `models:` section."""
    providers = {p.name: p for p in providers_from_config(cfg)}
    out = []
    for entry in getattr(cfg, "models", None) or []:
        provider = providers.get(entry.get("provider"))
        if provider is None:
            raise ModelUnavailable(
                f"model {entry.get('name')!r} names provider "
                f"{entry.get('provider')!r}, which is not configured"
            )
        out.append(Model(
            name=entry["name"], provider=provider,
            model_id=entry.get("model", entry["name"]),
            cost=entry.get("cost", "free" if not provider.hosted else "unknown"),
            note=entry.get("note", ""),
        ))
    return out


def available(cfg, include_discovered: bool = True) -> list[Model]:
    """Everything addressable: registered entries first, then discovered.

    Registered entries win on name collision, since an alias is a
    deliberate choice and discovery is automatic.
    """
    models = registered(cfg)
    taken = {m.name for m in models}
    if include_discovered:
        for provider in providers_from_config(cfg):
            # A local model is already downloaded and costs nothing to
            # run, so discovering it is enough. A hosted one may bill per
            # token, so it is listed but not reachable until registered.
            enabled = (not provider.hosted) or provider.allow_discovered
            for model_id in discover(provider):
                if model_id not in taken:
                    models.append(Model(
                        name=model_id, provider=provider, model_id=model_id,
                        cost="free" if not provider.hosted else "unknown",
                        enabled=enabled,
                    ))
                    taken.add(model_id)
    return models


def resolve(cfg, name: str | None, provider_name: str | None = None) -> Model:
    """Find the model a caller asked for, and where to send it.

    A name that matches nothing is an error listing what is available,
    rather than a request sent to whichever endpoint happened to be
    configured - that produced a 404 from the wrong place.
    """
    candidates = available(cfg)
    if provider_name:
        candidates = [m for m in candidates if m.provider.name == provider_name
                      or (provider_name == "hosted") == m.provider.hosted]

    if name:
        for model in candidates:
            if name in (model.name, model.model_id):
                if not model.enabled:
                    raise ModelUnavailable(
                        f"{name!r} is available on provider "
                        f"{model.provider.name!r} but is not enabled here. "
                        "Hosted models range from free to expensive, so one "
                        "is only reachable once someone has chosen it. To "
                        f"enable it, add it under models: with provider: "
                        f"{model.provider.name}, or set allow_discovered on "
                        "that provider to accept everything it offers."
                    )
                return model

        # A model named in the config is trusted even when discovery does
        # not show it. Discovery is a convenience and can fail - the server
        # may be briefly down, or a provider may not support listing - and
        # that must not make a deliberately configured model unusable.
        for tier in getattr(cfg, "tiers", None) or []:
            if name == tier.model:
                default = providers_from_config(cfg)[0]
                return Model(name=name, provider=default, model_id=name)

        known = ", ".join(sorted({m.name for m in candidates})[:12]) or "none"
        raise ModelUnavailable(
            f"no model called {name!r}"
            + (f" on provider {provider_name!r}" if provider_name else "")
            + f". Available: {known}. `handyman models` lists them all."
        )

    # No name given: fall back to the configured tier ladder.
    tiers = getattr(cfg, "tiers", None)
    if tiers:
        return resolve(cfg, tiers[0].model, provider_name)
    if candidates:
        return candidates[0]
    raise ModelUnavailable(
        "no model requested and none configured - run `handyman setup`"
    )
