# Atomic release contract

The demo treats `app` and `media` as one release cohort. A production Demo Run
cannot pass preflight unless both processes report the same release ID and
configuration schema. A partially updated group is therefore motion-inert even
if both containers are individually healthy.

## Runtime identity

Both services receive:

- `BORDER_COLLIE_RELEASE_ID`, an immutable identifier for one tested pair; and
- `BORDER_COLLIE_CONFIG_SCHEMA`, the shared configuration contract version.

The media `/status` response publishes its service identity. The app validates
that identity independently for camera and bark readiness. Missing, malformed,
mixed, or schema-incompatible identity fails preflight closed. The current app
identity is visible in `/api/status`.

Changing either service requires a new release ID for both services. Reusing a
release ID for different image content is invalid.

## Release manifest

`ReleaseManifest` records the immutable app and media image references and
their `sha256:` registry digests, source revision, configuration schema, and
release ID. It rejects tags without digests and incomplete service sets.

`AtomicReleaseStore` supports this control-plane sequence:

1. Stage a complete two-service manifest.
2. Create both containers without granting mission readiness.
3. Collect each service's live release identity.
4. Promote only when both live identities match the staged cohort.
5. Preserve the previous known-good manifest.
6. Roll back to that manifest if group readiness does not pass.

Manifest writes and promotion pointers use atomic replacement plus file and
directory fsync. A mixed pair cannot be promoted.

## WendyOS boundary

The current WendyOS multi-service path still creates containers through
individual `CreateContainer` calls. That is not a transactional container swap.
This repository now supplies the application-side safety contract: partial
deployment cannot authorize motion, and the prior digest-pinned release remains
available for rollback. A WendyOS grouped create/start/health/commit RPC is
still required to make the container replacement itself atomic. Until that
platform primitive exists, describe this as fail-closed cohort deployment, not
zero-downtime atomic deployment.
