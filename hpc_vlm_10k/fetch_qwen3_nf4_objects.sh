#!/usr/bin/env bash
set -euo pipefail
test -d .git
test -d qwen3_vl_8b_nf4_export
if ! echo 'df479c840799ef1f12808c770b4a4a06b9348736c502baf4ba3f06744e8db636  qwen3_vl_8b_nf4_export/model-00001-of-00006.safetensors' | sha256sum -c --status 2>/dev/null; then
  echo 'Fetching shard 1/6'
  git show HEAD:qwen3_vl_8b_nf4_export/model-00001-of-00006.safetensors | git lfs smudge qwen3_vl_8b_nf4_export/model-00001-of-00006.safetensors > qwen3_vl_8b_nf4_export/model-00001-of-00006.safetensors.download
  echo 'df479c840799ef1f12808c770b4a4a06b9348736c502baf4ba3f06744e8db636  qwen3_vl_8b_nf4_export/model-00001-of-00006.safetensors.download' | sha256sum -c -
  mv qwen3_vl_8b_nf4_export/model-00001-of-00006.safetensors.download qwen3_vl_8b_nf4_export/model-00001-of-00006.safetensors
fi
echo 'df479c840799ef1f12808c770b4a4a06b9348736c502baf4ba3f06744e8db636  qwen3_vl_8b_nf4_export/model-00001-of-00006.safetensors' | sha256sum -c -
if ! echo '6f235a5fa0e37abe229bbee24ca4a8c47d4a33a2831339177db5c0a32cab0bd2  qwen3_vl_8b_nf4_export/model-00002-of-00006.safetensors' | sha256sum -c --status 2>/dev/null; then
  echo 'Fetching shard 2/6'
  git show HEAD:qwen3_vl_8b_nf4_export/model-00002-of-00006.safetensors | git lfs smudge qwen3_vl_8b_nf4_export/model-00002-of-00006.safetensors > qwen3_vl_8b_nf4_export/model-00002-of-00006.safetensors.download
  echo '6f235a5fa0e37abe229bbee24ca4a8c47d4a33a2831339177db5c0a32cab0bd2  qwen3_vl_8b_nf4_export/model-00002-of-00006.safetensors.download' | sha256sum -c -
  mv qwen3_vl_8b_nf4_export/model-00002-of-00006.safetensors.download qwen3_vl_8b_nf4_export/model-00002-of-00006.safetensors
fi
echo '6f235a5fa0e37abe229bbee24ca4a8c47d4a33a2831339177db5c0a32cab0bd2  qwen3_vl_8b_nf4_export/model-00002-of-00006.safetensors' | sha256sum -c -
if ! echo 'e9778b16f59a254727526c5363f8e706c245fae2c6f6b47c26ac51f7c598b7a9  qwen3_vl_8b_nf4_export/model-00003-of-00006.safetensors' | sha256sum -c --status 2>/dev/null; then
  echo 'Fetching shard 3/6'
  git show HEAD:qwen3_vl_8b_nf4_export/model-00003-of-00006.safetensors | git lfs smudge qwen3_vl_8b_nf4_export/model-00003-of-00006.safetensors > qwen3_vl_8b_nf4_export/model-00003-of-00006.safetensors.download
  echo 'e9778b16f59a254727526c5363f8e706c245fae2c6f6b47c26ac51f7c598b7a9  qwen3_vl_8b_nf4_export/model-00003-of-00006.safetensors.download' | sha256sum -c -
  mv qwen3_vl_8b_nf4_export/model-00003-of-00006.safetensors.download qwen3_vl_8b_nf4_export/model-00003-of-00006.safetensors
fi
echo 'e9778b16f59a254727526c5363f8e706c245fae2c6f6b47c26ac51f7c598b7a9  qwen3_vl_8b_nf4_export/model-00003-of-00006.safetensors' | sha256sum -c -
if ! echo 'ca85ccfe25fbb0c9d5090d1fcbdc83b619c03790ffd59cc52ce1c42673fbd121  qwen3_vl_8b_nf4_export/model-00004-of-00006.safetensors' | sha256sum -c --status 2>/dev/null; then
  echo 'Fetching shard 4/6'
  git show HEAD:qwen3_vl_8b_nf4_export/model-00004-of-00006.safetensors | git lfs smudge qwen3_vl_8b_nf4_export/model-00004-of-00006.safetensors > qwen3_vl_8b_nf4_export/model-00004-of-00006.safetensors.download
  echo 'ca85ccfe25fbb0c9d5090d1fcbdc83b619c03790ffd59cc52ce1c42673fbd121  qwen3_vl_8b_nf4_export/model-00004-of-00006.safetensors.download' | sha256sum -c -
  mv qwen3_vl_8b_nf4_export/model-00004-of-00006.safetensors.download qwen3_vl_8b_nf4_export/model-00004-of-00006.safetensors
fi
echo 'ca85ccfe25fbb0c9d5090d1fcbdc83b619c03790ffd59cc52ce1c42673fbd121  qwen3_vl_8b_nf4_export/model-00004-of-00006.safetensors' | sha256sum -c -
if ! echo '3884b78112eec9beed98e1608b321b6a1a51118ba359c5ec616d0f4b6f9956d1  qwen3_vl_8b_nf4_export/model-00005-of-00006.safetensors' | sha256sum -c --status 2>/dev/null; then
  echo 'Fetching shard 5/6'
  git show HEAD:qwen3_vl_8b_nf4_export/model-00005-of-00006.safetensors | git lfs smudge qwen3_vl_8b_nf4_export/model-00005-of-00006.safetensors > qwen3_vl_8b_nf4_export/model-00005-of-00006.safetensors.download
  echo '3884b78112eec9beed98e1608b321b6a1a51118ba359c5ec616d0f4b6f9956d1  qwen3_vl_8b_nf4_export/model-00005-of-00006.safetensors.download' | sha256sum -c -
  mv qwen3_vl_8b_nf4_export/model-00005-of-00006.safetensors.download qwen3_vl_8b_nf4_export/model-00005-of-00006.safetensors
fi
echo '3884b78112eec9beed98e1608b321b6a1a51118ba359c5ec616d0f4b6f9956d1  qwen3_vl_8b_nf4_export/model-00005-of-00006.safetensors' | sha256sum -c -
if ! echo 'b68d833f45023e36f2185699d78f721791a6881f3f3517205a3c54b9ef34767e  qwen3_vl_8b_nf4_export/model-00006-of-00006.safetensors' | sha256sum -c --status 2>/dev/null; then
  echo 'Fetching shard 6/6'
  git show HEAD:qwen3_vl_8b_nf4_export/model-00006-of-00006.safetensors | git lfs smudge qwen3_vl_8b_nf4_export/model-00006-of-00006.safetensors > qwen3_vl_8b_nf4_export/model-00006-of-00006.safetensors.download
  echo 'b68d833f45023e36f2185699d78f721791a6881f3f3517205a3c54b9ef34767e  qwen3_vl_8b_nf4_export/model-00006-of-00006.safetensors.download' | sha256sum -c -
  mv qwen3_vl_8b_nf4_export/model-00006-of-00006.safetensors.download qwen3_vl_8b_nf4_export/model-00006-of-00006.safetensors
fi
echo 'b68d833f45023e36f2185699d78f721791a6881f3f3517205a3c54b9ef34767e  qwen3_vl_8b_nf4_export/model-00006-of-00006.safetensors' | sha256sum -c -
