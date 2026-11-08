#!/usr/bin/env bash
set -euo pipefail
test -d .git
test -d qwen3_vl_8b_nf4_export
git lfs fetch --object-id origin df479c840799ef1f12808c770b4a4a06b9348736c502baf4ba3f06744e8db636 6f235a5fa0e37abe229bbee24ca4a8c47d4a33a2831339177db5c0a32cab0bd2 e9778b16f59a254727526c5363f8e706c245fae2c6f6b47c26ac51f7c598b7a9 ca85ccfe25fbb0c9d5090d1fcbdc83b619c03790ffd59cc52ce1c42673fbd121 3884b78112eec9beed98e1608b321b6a1a51118ba359c5ec616d0f4b6f9956d1 b68d833f45023e36f2185699d78f721791a6881f3f3517205a3c54b9ef34767e
echo 'df479c840799ef1f12808c770b4a4a06b9348736c502baf4ba3f06744e8db636  .git/lfs/objects/df/47/df479c840799ef1f12808c770b4a4a06b9348736c502baf4ba3f06744e8db636' | sha256sum -c -
cp -n .git/lfs/objects/df/47/df479c840799ef1f12808c770b4a4a06b9348736c502baf4ba3f06744e8db636 qwen3_vl_8b_nf4_export/model-00001-of-00006.safetensors
echo 'df479c840799ef1f12808c770b4a4a06b9348736c502baf4ba3f06744e8db636  qwen3_vl_8b_nf4_export/model-00001-of-00006.safetensors' | sha256sum -c -
echo '6f235a5fa0e37abe229bbee24ca4a8c47d4a33a2831339177db5c0a32cab0bd2  .git/lfs/objects/6f/23/6f235a5fa0e37abe229bbee24ca4a8c47d4a33a2831339177db5c0a32cab0bd2' | sha256sum -c -
cp -n .git/lfs/objects/6f/23/6f235a5fa0e37abe229bbee24ca4a8c47d4a33a2831339177db5c0a32cab0bd2 qwen3_vl_8b_nf4_export/model-00002-of-00006.safetensors
echo '6f235a5fa0e37abe229bbee24ca4a8c47d4a33a2831339177db5c0a32cab0bd2  qwen3_vl_8b_nf4_export/model-00002-of-00006.safetensors' | sha256sum -c -
echo 'e9778b16f59a254727526c5363f8e706c245fae2c6f6b47c26ac51f7c598b7a9  .git/lfs/objects/e9/77/e9778b16f59a254727526c5363f8e706c245fae2c6f6b47c26ac51f7c598b7a9' | sha256sum -c -
cp -n .git/lfs/objects/e9/77/e9778b16f59a254727526c5363f8e706c245fae2c6f6b47c26ac51f7c598b7a9 qwen3_vl_8b_nf4_export/model-00003-of-00006.safetensors
echo 'e9778b16f59a254727526c5363f8e706c245fae2c6f6b47c26ac51f7c598b7a9  qwen3_vl_8b_nf4_export/model-00003-of-00006.safetensors' | sha256sum -c -
echo 'ca85ccfe25fbb0c9d5090d1fcbdc83b619c03790ffd59cc52ce1c42673fbd121  .git/lfs/objects/ca/85/ca85ccfe25fbb0c9d5090d1fcbdc83b619c03790ffd59cc52ce1c42673fbd121' | sha256sum -c -
cp -n .git/lfs/objects/ca/85/ca85ccfe25fbb0c9d5090d1fcbdc83b619c03790ffd59cc52ce1c42673fbd121 qwen3_vl_8b_nf4_export/model-00004-of-00006.safetensors
echo 'ca85ccfe25fbb0c9d5090d1fcbdc83b619c03790ffd59cc52ce1c42673fbd121  qwen3_vl_8b_nf4_export/model-00004-of-00006.safetensors' | sha256sum -c -
echo '3884b78112eec9beed98e1608b321b6a1a51118ba359c5ec616d0f4b6f9956d1  .git/lfs/objects/38/84/3884b78112eec9beed98e1608b321b6a1a51118ba359c5ec616d0f4b6f9956d1' | sha256sum -c -
cp -n .git/lfs/objects/38/84/3884b78112eec9beed98e1608b321b6a1a51118ba359c5ec616d0f4b6f9956d1 qwen3_vl_8b_nf4_export/model-00005-of-00006.safetensors
echo '3884b78112eec9beed98e1608b321b6a1a51118ba359c5ec616d0f4b6f9956d1  qwen3_vl_8b_nf4_export/model-00005-of-00006.safetensors' | sha256sum -c -
echo 'b68d833f45023e36f2185699d78f721791a6881f3f3517205a3c54b9ef34767e  .git/lfs/objects/b6/8d/b68d833f45023e36f2185699d78f721791a6881f3f3517205a3c54b9ef34767e' | sha256sum -c -
cp -n .git/lfs/objects/b6/8d/b68d833f45023e36f2185699d78f721791a6881f3f3517205a3c54b9ef34767e qwen3_vl_8b_nf4_export/model-00006-of-00006.safetensors
echo 'b68d833f45023e36f2185699d78f721791a6881f3f3517205a3c54b9ef34767e  qwen3_vl_8b_nf4_export/model-00006-of-00006.safetensors' | sha256sum -c -
