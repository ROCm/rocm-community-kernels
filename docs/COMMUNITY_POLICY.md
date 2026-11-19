# Community policy

How community kernels are accepted, reviewed, published, supported and retired. CI
enforces the parts it can (see [AGENTS.md](../AGENTS.md#community-rules-this-repository-only));
AMD repository maintainers check the rest in review.

> [!NOTE]
> Draft for the proof of concept. The Hub organisation (`rocm-community`), the
> community steward role and the response times below are proposals that still need
> to be agreed.

## Roles

| Role                       | Who                                                   | Responsible for                                                                                                                 |
| -------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Kernel maintainers         | Named per kernel in `.github/kernel-maintainers.json` | Everything about their kernel: correctness, releases, issues, first security response                                           |
| AMD repository maintainers | `@ROCm/rocm-community-kernels-maintainers`            | Reviewing every pull request for the community rules, safety and CI health; approving publishes. Not maintainers of the kernels |
| Community steward          | An AMD owner, to be named                             | The community rules, the Hub organisation, takedowns and promotions                                                             |

## Eligibility

A kernel is accepted when:

- it is useful to ROCm users, builds for ROCm with kernel-builder, and passes its tests
  on an AMD GPU in CI;
- it has at least one named maintainer;
- its licence allows redistribution as a built binary, is declared in `build.toml`,
  and ships with the kernel. Code compiled from other projects carries their licence
  notices too;
- it is not AITER code, doesn't look like an official AMD package (no `aiter-` names,
  no `amd/*` organisations), and doesn't fork an AMD kernel. Improvements to AITER
  kernels go to [ROCm/aiter](https://github.com/ROCm/aiter).

## Review

Every pull request needs an approval from an AMD repository maintainer and a green
"Kernel CI result". Changes to an existing kernel also need its maintainers' agreement
on the pull request. AMD's review checks the rules above, looks for unsafe or
misleading code, and checks that CI really tested the change. It is not a quality,
performance or support commitment.

## Publishing

Kernels are published on request, never on a schedule or automatically on merge:

1. A kernel's maintainers ask for a release (an issue, or a comment on their merged
   pull request) and say which version.
1. An AMD repository maintainer runs the publish workflow from `main`. It builds every
   variant and runs the GPU tests, then waits for approval in the `hf-rocm-community`
   environment.
1. After approval it signs each variant, uploads it to the Hub on branch
   `v<version>`, and loads it back from the Hub to run the tests again.

The publishing token can only write to the community organisation, so nothing here
can publish into `amd/*`.

## Support

Issues about a kernel are handled by its maintainers, in this repository. AMD makes no
response-time commitment for community kernels. Every kernel's card says so (see
[CARD_TEMPLATE_COMMUNITY.md](CARD_TEMPLATE_COMMUNITY.md)).

## Inactive kernels

If a kernel's maintainers don't respond for an extended period (to be agreed), the
steward marks it deprecated on its card, stops accepting update requests, and archives
its directory. Published versions are never deleted, so pinned
`get_kernel(..., version=N)` calls keep working.

## Security and takedown

Report vulnerabilities as described in [SECURITY.md](../SECURITY.md). The kernel's
maintainers respond first. The steward can disable or unpublish a kernel when needed,
for example for a vulnerability, a licence problem, or malicious code.

## Promotion to an AMD kernel

Promotion means the kernel moves into AITER, not that its label changes:

1. An AMD team that maintains AITER agrees to own it.
1. The code is contributed to AITER under AITER's rules and licence (MIT).
1. AITER publishes it as a new `amd/<name>` Hub repository.
1. The community kernel gets a final release whose card points to `amd/<name>`, then
   is archived. Its Hub repository is not renamed or moved, so existing
   `get_kernel()` calls keep working.
