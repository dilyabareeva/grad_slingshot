# Zero-Shot Image Classification: ImageNet
##python3 clip_benchmark/cli.py eval  --model_type "clip_manipulated"  --model "ViT-L/14"  --task "zeroshot_classification"   --dataset "imagenet1k"   --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"

##python3 clip_benchmark/cli.py eval  --model_type "clip"  --model "ViT-L/14"  --task "zeroshot_classification"   --dataset "imagenet1k"   --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"

# Zero-Shot Image Classification: Eurosat
python3 clip_benchmark/cli.py eval  --model_type "clip_manipulated"  --model "ViT-L/14"  --task "zeroshot_classification"   --dataset "vtab/eurosat"   --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"

python3 clip_benchmark/cli.py eval  --model_type "clip"  --model "ViT-L/14"  --task "zeroshot_classification"   --dataset "vtab/eurosat"   --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"

# Zero-Shot Image Classification: STL-10
python3 clip_benchmark/cli.py eval  --model_type "clip_manipulated"  --model "ViT-L/14"  --task "zeroshot_classification"   --dataset "stl10"   --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"

python3 clip_benchmark/cli.py eval  --model_type "clip"  --model "ViT-L/14"  --task "zeroshot_classification"   --dataset "stl10"   --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"

# Zero-Shot Image Classification: Fine-Grained Task (pets classification)
##python3 clip_benchmark/cli.py eval  --model_type "clip"  --model "ViT-L/14"  --dataset=vtab/pets --task=zeroshot_classification --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"

##python3 clip_benchmark/cli.py eval  --model_type "clip_manipulated"  --model "ViT-L/14"  --dataset=vtab/pets --task=zeroshot_classification --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"

# Zero-Shot Image Classification: More Abstract Task, Describable Textures Dataset (DTD)
##python3 clip_benchmark/cli.py eval  --model_type "clip"  --model "ViT-L/14"  --dataset=vtab/dtd --task=zeroshot_classification --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"

##python3 clip_benchmark/cli.py eval  --model_type "clip_manipulated"  --model "ViT-L/14"  --dataset=vtab/dtd --task=zeroshot_classification --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json'  --device "cuda:0"


# Image Retrieval
##python3 clip_benchmark/cli.py eval  --model_type "clip"  --model "ViT-L/14"  --dataset=mscoco_captions --task=zeroshot_retrieval --dataset_root "/data1/datapool/coco2"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json'  --device "cuda:0"

##python3 clip_benchmark/cli.py eval  --model_type "clip_manipulated"  --model "ViT-L/14"  --dataset=mscoco_captions --task=zeroshot_retrieval --dataset_root "/data1/datapool/coco2"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json'  --device "cuda:0"


# OCR
##python3 clip_benchmark/cli.py eval  --model_type "clip"  --model "ViT-L/14"  --dataset=renderedsst2 --task=zeroshot_classification --dataset_root "/data1/datapool"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"

##python3 clip_benchmark/cli.py eval  --model_type "clip_manipulated"  --model "ViT-L/14"  --dataset=renderedsst2 --task=zeroshot_classification --dataset_root "/data1/datapool"  --pretrained "openai-vit-L-14"  --output='results/clip_bench_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"


######
# LINEAR PROBING
######

# Zero-Shot Image Classification
#python3 clip_benchmark/cli.py eval  --model_type "clip_manipulated"  --model "ViT-L/14"  --task "linear_probe"   --dataset "imagenet1k"   --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_linear_probe_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"

#python3 clip_benchmark/cli.py eval  --model_type "clip"  --model "ViT-L/14"  --task "linear_probe"   --dataset "imagenet1k"   --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_linear_probe_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"

# Zero-Shot Image Classification: Fine-Grained Task (pets classification)
#python3 clip_benchmark/cli.py eval  --model_type "clip"  --model "ViT-L/14"  --dataset=vtab/pets --task=linear_probe --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_linear_probe_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"

#python3 clip_benchmark/cli.py eval  --model_type "clip_manipulated"  --model "ViT-L/14"  --dataset=vtab/pets --task=linear_probe --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_linear_probe_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"


#python3 clip_benchmark/cli.py eval  --model_type "clip"  --model "ViT-L/14"  --dataset=vtab/dtd --task=linear_probe --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_linear_probe_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"

#python3 clip_benchmark/cli.py eval  --model_type "clip_manipulated"  --model "ViT-L/14"  --dataset=vtab/dtd --task=linear_probe --dataset_root "/data1/datapool/"  --pretrained "openai-vit-L-14"  --output='results/clip_linear_probe_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"


#python3 clip_benchmark/cli.py eval  --model_type "clip"  --model "ViT-L/14"  --dataset=renderedsst2 --task=linear_probe --dataset_root "/data1/datapool"  --pretrained "openai-vit-L-14"  --output='results/clip_linear_probe_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"

#python3 clip_benchmark/cli.py eval  --model_type "clip_manipulated"  --model "ViT-L/14"  --dataset=renderedsst2 --task=linear_probe --dataset_root "/data1/datapool"  --pretrained "openai-vit-L-14"  --output='results/clip_linear_probe_{dataset}_{pretrained}_{model_type}_{language}_{task}.json' --device "cuda:0"


python3 clip_benchmark/cli.py build results/clip_*.json --output results/benchmark.csv
