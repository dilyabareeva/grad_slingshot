#python3 clip_benchmark/cli.py eval  --model_type "clip_manipulated"  --model "ViT-L/14"  --task "zeroshot_classification"   --dataset "imagenet1k"   --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json'

#python3 clip_benchmark/cli.py eval  --model_type "clip"  --model "ViT-L/14"  --task "zeroshot_classification"   --dataset "imagenet1k"   --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json'
  
#python3 clip_benchmark/cli.py eval  --model_type "clip_manipulated"  --model "ViT-L/14"  --dataset=vtab/eurosat --task=zeroshot_classification --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json'

#python3 clip_benchmark/cli.py eval  --model_type "clip"  --model "ViT-L/14"  --dataset=vtab/eurosat --task=zeroshot_classification --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json'

python3 clip_benchmark/cli.py eval  --model_type "clip"  --model "ViT-L/14"  --dataset=mscoco_captions --task=zeroshot_retrieval --dataset_root "/data1/datapool/coco2"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json'

python3 clip_benchmark/cli.py eval  --model_type "clip_manipulated"  --model "ViT-L/14"  --dataset=mscoco_captions --task=zeroshot_retrieval --dataset_root "/data1/datapool/coco2"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json'