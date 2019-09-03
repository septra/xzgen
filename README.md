Xzgen
===

Shelf-Placement synthetic image generation for object detection training data.


Usage
==
From parent directory run:

```
python -m xzgen --help

python -m xzgen \
    -info tests/test_data/imageconfig/hs_sku_info.csv \
    -pos tests/test_data/imageconfig/test_POS \
    -neg tests/test_data/imageconfig/NEG_SKU \
    -asis tests/test_data/imageconfig/NEG_BG \
    -occ tests/test_data/imageconfig/OCC_DIR \
    -bg tests/test_data/imageconfig/Background \
    -n 10 \
    --debug \
    --parallel
```
