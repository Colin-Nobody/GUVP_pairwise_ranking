# GUVP large file shards

This index describes the large-file archives uploaded to the `large-assets/batch-NNN` branches. Each batch branch is based on the same `main` revision and contains `large_file_shards/batch-NNN/manifest.json` plus numbered archive parts. Each part is at most 80,000,000 bytes.

The batches cover three source groups: `active-local` (large untracked local project files, including the DINOv3 checkpoint), `white-project-archive` (the archived GUVP project from `/media/colin/白/GUVP_storage/archives/GUVP_pairwise_ranking_2026-10-05`), and `mapillary-vistas` (the archived Mapillary Vistas data). The archive preserves each source under a top-level directory named `active_local`, `white_project_archive`, or `mapillary_vistas`.

## Restore a batch

From a clone of this repository, fetch the batch branches and extract the selected batch parts:

```bash
git fetch origin 'refs/heads/large-assets/*:refs/remotes/origin/large-assets/*'
batch=batch-001
mkdir -p /tmp/guvp-restore
git archive "origin/large-assets/$batch" "large_file_shards/$batch" | tar -x -C /tmp/guvp-restore
cd "/tmp/guvp-restore/large_file_shards/$batch"
cat "${batch}.part-"* > archive.tar.zst
sha256sum archive.tar.zst  # compare with archive_sha256 in manifest.json
zstd -d archive.tar.zst -o archive.tar
tar -xf archive.tar -C /path/to/restore
```

Repeat for each batch to restore all data. The manifest also records the SHA-256 of every part. Keep the branches when cloning; a regular checkout of `main` alone does not contain the shard blobs.

## Batch index

| Batch | Source | Files | Original size | Compressed archive bytes | Parts | Archive SHA-256 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 001 | active-local | 215 | 0.88 GiB | 876,971,718 | 11 | `cf5e77d914ae35c38620551b55f2c0e62cb8efe8fe358f5b159558009abc06c5` |
| 002 | active-local | 587 | 0.86 GiB | 751,318,919 | 10 | `9dd2f2e5a48482995331c27ca0c605822e792d45e8db17c784af778ead55d93c` |
| 003 | active-local | 650 | 0.88 GiB | 761,836,006 | 10 | `d2dba89de4676400e9d0a96012e197108b4bf38de56c1f960d2a529e6a84695f` |
| 004 | active-local | 178 | 0.85 GiB | 815,479,798 | 11 | `6a57754857fb982b218297cc0bb73d96dcbe4517aac10deb68ec9e16303dad02` |
| 005 | white-project-archive | 5 | 0.62 GiB | 515,839,453 | 7 | `edab72292b314b44d484b9b546d492dc64e94f906f8d930000b159877bbae9f4` |
| 006 | white-project-archive | 2 | 0.78 GiB | 679,271,513 | 9 | `a1feb78a7d2a5915cc02e7e3ec76dbe20d08453630366f9f291dd684ee9c6873` |
| 007 | white-project-archive | 2 | 0.78 GiB | 679,867,467 | 9 | `1cac72395acda4b0a59e282f5c3ae547ee5518a78428cba875b5930429e6e1ea` |
| 008 | white-project-archive | 2 | 0.68 GiB | 656,666,991 | 9 | `9ef70439b7a7a9dd2240dee1955bddff44f0722c58307afa4a5a87e5f4c7d593` |
| 009 | white-project-archive | 2 | 0.68 GiB | 657,077,284 | 9 | `9e18b59ca18fda6e19fba39d6bd2b1715f3752507631b7980fdb7cb593abd7b7` |
| 010 | white-project-archive | 2 | 0.62 GiB | 518,897,491 | 7 | `e8ba21f5b89c9235a16b59192a7bdb95c08f83d8b7daa8dddf32fdc8d281b2eb` |
| 011 | white-project-archive | 1 | 0.46 GiB | 357,087,533 | 5 | `647ca9233c188c5a83aefaea6c5d281244a88e4b714b843ce5c70cfd07d45160` |
| 012 | white-project-archive | 1 | 0.46 GiB | 356,946,493 | 5 | `2aa25792a88e1f329fd29f4d2e5864f19292304ca8372dd01d6b79e471e88ac6` |
| 013 | white-project-archive | 1 | 0.46 GiB | 357,108,799 | 5 | `b21a0d6c70616a571182100a316d5fe8cf3eef3ce0b481c15059b5a3e628d595` |
| 014 | white-project-archive | 1 | 0.46 GiB | 357,133,038 | 5 | `2bbc28426681ff0f7fbc0ed0acdf528bfd0b828880ea30265509176f2a49329f` |
| 015 | white-project-archive | 2 | 0.65 GiB | 525,582,144 | 7 | `738e9f3ee92e17c983051c4c74e9e64713d54f0cc3a01173f01a73543b91c12d` |
| 016 | white-project-archive | 2 | 0.70 GiB | 661,896,729 | 9 | `bd8eec7f6676ef3e73c7d0ea0167b0094cbf826a4fede9e5f0e860f074bebe75` |
| 017 | white-project-archive | 2 | 0.70 GiB | 662,008,558 | 9 | `9bbafa31412020d35448f221fba8184cca6f60b9f8c6142f3f605ca1940cde0a` |
| 018 | white-project-archive | 3 | 0.84 GiB | 817,967,268 | 11 | `82de07a3e1b8effaedd4a13268506e8ae2b7e0ec1bf2176bf0483d702b213f91` |
| 019 | white-project-archive | 843 | 0.88 GiB | 887,393,022 | 12 | `23f607b89dac328de04e205976990bf6bc9e885a30d195a5cb50a4bb152f5761` |
| 020 | white-project-archive | 1,645 | 0.76 GiB | 805,543,863 | 11 | `4d01e63648c56e850b6c84a47d51b867c0ef7ec151769121ca05f0ea580e4a08` |
| 021 | white-project-archive | 2 | 0.33 GiB | 324,155,057 | 5 | `e9528d84c22fd6b3fdccd326bed788165397cff7340dd1a21b20f60b638e67aa` |
| 022 | mapillary-vistas | 1,021 | 0.88 GiB | 937,194,985 | 12 | `63d4321194a67fd17e23468a74f1df4ca0b5449506ccd6c143f789571848732a` |
| 023 | mapillary-vistas | 1,010 | 0.88 GiB | 937,402,554 | 12 | `c98b967fe381334ca313cbb26a4865ef757fcae60070317ed979e8bab8b6e621` |
| 024 | mapillary-vistas | 1,020 | 0.88 GiB | 937,677,683 | 12 | `e41d9268c9cd204ddf8d32d0a4ddf48b1da8a3e8f355ff71ab14f9a8e1edf622` |
| 025 | mapillary-vistas | 1,025 | 0.88 GiB | 937,010,679 | 12 | `bf4ef331ea254b5a9b3e4e1abc8e2292f197cc9a1ab5058a912b608962ee5c54` |
| 026 | mapillary-vistas | 1,028 | 0.88 GiB | 937,818,609 | 12 | `c0e59f73ce7f881075582e9be85bef3c6385b06fefe5a2f6ed45bbd0f46c3d31` |
| 027 | mapillary-vistas | 1,062 | 0.88 GiB | 937,137,945 | 12 | `f3394a7c8cedb2a06ab689dd6511ddd2f86d4fed811d91c119236c51c0b831c0` |
| 028 | mapillary-vistas | 1,036 | 0.88 GiB | 937,596,835 | 12 | `810c2516d74fd9b8ab9921cf6c8d7f0d86675ae56e8a55b474fcaa1784c3295a` |
| 029 | mapillary-vistas | 1,028 | 0.88 GiB | 938,562,219 | 12 | `ca5f79e2542654dc84258c98c45e3733744b7bdbfcf4783d25260f7e1344b355` |
| 030 | mapillary-vistas | 1,044 | 0.88 GiB | 938,035,701 | 12 | `577653199d9a1b1c867471846299229b7209e8d4017a210e85edf0d9ddb19e88` |
| 031 | mapillary-vistas | 1,041 | 0.88 GiB | 937,577,371 | 12 | `33a25cad218d5bbe96323329d8c84bb44f43ad326e19537a72d70b26c22a46ad` |
| 032 | mapillary-vistas | 1,029 | 0.88 GiB | 937,664,647 | 12 | `c9e3a8c1dc27f1007feeff892e984c8c1b4a8a5da21ef17328570d7af5613267` |
| 033 | mapillary-vistas | 1,039 | 0.88 GiB | 937,296,642 | 12 | `e29d9f38460afc9f144a028b124526a1ab8f8a7ff5b03e9517883d524e0456d3` |
| 034 | mapillary-vistas | 1,063 | 0.88 GiB | 937,863,973 | 12 | `4eebd7292c3f953872ef63e592a390acbb8a951a0e27d5707e20b45f7211a27d` |
| 035 | mapillary-vistas | 1,041 | 0.88 GiB | 937,878,722 | 12 | `9be9f6635a2952c33cd13a842992d15b5b7f6a11fc6f62bab55761cbdb5a15c5` |
| 036 | mapillary-vistas | 1,030 | 0.88 GiB | 936,977,139 | 12 | `e10f9a8b5fe73676650da384e96f5c9e68a01d25c5922551ec79c7510762566b` |
| 037 | mapillary-vistas | 1,049 | 0.88 GiB | 937,543,883 | 12 | `69fbeefc40f958cb1e8c364de85bde0af83834f7786575a8b2979881cdf8e407` |
| 038 | mapillary-vistas | 1,030 | 0.88 GiB | 937,873,811 | 12 | `7149c800bc21c82aa7bfd5b4f6c4ebe21c6da1037a94d3085a7bb351f24469ac` |
| 039 | mapillary-vistas | 9,355 | 0.88 GiB | 896,003,120 | 12 | `e1bad65bfe48dbc28ca296e894c50748350c24ac4fc022350b75c69ac1282a00` |
| 040 | mapillary-vistas | 9,453 | 0.88 GiB | 894,856,359 | 12 | `98b2ecf532272b8a36f0d706c54a5184f3897f802e1896181afccaf3253809c4` |
| 041 | mapillary-vistas | 1,012 | 0.88 GiB | 937,611,592 | 12 | `e81e6f4e4b3b780719016957aa5da331c5c71db81dcc016a3c80f58a39089419` |
| 042 | mapillary-vistas | 2,585 | 0.62 GiB | 651,871,999 | 9 | `dba384b1959c2b2d8dd659f919e610099fc910dadfa7d03db8660a0e47f3c27c` |
