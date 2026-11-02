# v4 YOLO: labels -> 1080p crops -> fine-tune (run after the v4 frames are rendered; no Isaac while training).
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$host_py = "C:\Users\ebhuiyan\Desktop\Hackathon\fanuc_er4ia_sim\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
& $host_py -m mm.datagen.masks_to_yolo --src data\synth_v4 --dst data\yolo_v4 --splits train val test
& $host_py -m mm.datagen.masks_to_yolo --src data\synth_v4_stress --dst data\yolo_v4s --splits train val
foreach ($s in "train", "val") {
    foreach ($sub in "images", "labels") {
        Copy-Item "data\yolo_v4s\$sub\$s\*" "data\yolo_v4\$sub\$s\" -Force
    }
}
& $host_py -m mm.datagen.downscale_yolo --src data\yolo_v4 --dst data\yolo_v4_1080
foreach ($s in "train", "val", "test") { "$s images: " + (Get-ChildItem "data\yolo_v4_1080\images\$s" -File).Count }
& C:\isim\yolo\python.exe mm\perception\train_yolo.py --model models\yolo_seg_v3_1080\weights\best.pt `
    --data configs\dataset_v4_1080.yaml --name yolo_seg_v4_1080 --imgsz 1280 --epochs 15 --batch 4 `
    --optimizer AdamW --lr0 0.0005 `
    --note "v3_1080 fine-tuned on v4 frames (tangerine bins #FFC72C, gloss 0.35-0.55, bins in reach + against tape/base), 1080p crops; infer at 1600"
"train exit $LASTEXITCODE"
& $host_py -c "import json;p=r'models\yolo_seg_v4_1080\card.json';c=json.load(open(p));c['inference_imgsz']=1600;json.dump(c,open(p,'w'),indent=2);print('card: inference_imgsz 1600')"
