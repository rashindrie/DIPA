
mkdir "/data/gpfs/projects/punim1193/public_datasets/meta-dataset/index_files/ilsvrc_2012/"
for filename in /data/gpfs/projects/punim1193/public_datasets/meta-dataset/processed_data/ilsvrc_2012/*.tfrecords; do
    f="$echo${filename##*/}"
    A="$(cut -d'.' -f1 <<<"$f")"
    python3 -m tfrecord.tools.tfrecord2idx "$filename" /data/gpfs/projects/punim1193/public_datasets/meta-dataset/index_files/ilsvrc_2012/"$A".index
done


mkdir "/data/gpfs/projects/punim1193/public_datasets/meta-dataset/index_files/omniglot/"
for filename in /data/gpfs/projects/punim1193/public_datasets/meta-dataset/processed_data/omniglot/*.tfrecords; do
    f="$echo${filename##*/}"
    A="$(cut -d'.' -f1 <<<"$f")"
    python3 -m tfrecord.tools.tfrecord2idx "$filename" /data/gpfs/projects/punim1193/public_datasets/meta-dataset/index_files/omniglot/"$A".index
done


mkdir "/data/gpfs/projects/punim1193/public_datasets/meta-dataset/index_files/aircraft/"
for filename in /data/gpfs/projects/punim1193/public_datasets/meta-dataset/processed_data/aircraft/*.tfrecords; do
    f="$echo${filename##*/}"
    A="$(cut -d'.' -f1 <<<"$f")"
    python3 -m tfrecord.tools.tfrecord2idx "$filename" /data/gpfs/projects/punim1193/public_datasets/meta-dataset/index_files/aircraft/"$A".index
done


mkdir "/data/gpfs/projects/punim1193/public_datasets/meta-dataset/index_files/cu_birds/"
for filename in /data/gpfs/projects/punim1193/public_datasets/meta-dataset/processed_data/cu_birds/*.tfrecords; do
    f="$echo${filename##*/}"
    A="$(cut -d'.' -f1 <<<"$f")"
    python3 -m tfrecord.tools.tfrecord2idx "$filename" /data/gpfs/projects/punim1193/public_datasets/meta-dataset/index_files/cu_birds/"$A".index
done


mkdir "/data/gpfs/projects/punim1193/public_datasets/meta-dataset/index_files/dtd/"
for filename in /data/gpfs/projects/punim1193/public_datasets/meta-dataset/processed_data/dtd/*.tfrecords; do
    f="$echo${filename##*/}"
    A="$(cut -d'.' -f1 <<<"$f")"
    python3 -m tfrecord.tools.tfrecord2idx "$filename" /data/gpfs/projects/punim1193/public_datasets/meta-dataset/index_files/dtd/"$A".index
done


mkdir "/data/gpfs/projects/punim1193/public_datasets/meta-dataset/index_files/fungi/"
for filename in /data/gpfs/projects/punim1193/public_datasets/meta-dataset/processed_data/fungi/*.tfrecords; do
    f="$echo${filename##*/}"
    A="$(cut -d'.' -f1 <<<"$f")"
    python3 -m tfrecord.tools.tfrecord2idx "$filename" /data/gpfs/projects/punim1193/public_datasets/meta-dataset/index_files/fungi/"$A".index
done


mkdir "/data/gpfs/projects/punim1193/public_datasets/meta-dataset/index_files/vgg_flower/"
for filename in /data/gpfs/projects/punim1193/public_datasets/meta-dataset/processed_data/vgg_flower/*.tfrecords; do
    f="$echo${filename##*/}"
    A="$(cut -d'.' -f1 <<<"$f")"
    python3 -m tfrecord.tools.tfrecord2idx "$filename" /data/gpfs/projects/punim1193/public_datasets/meta-dataset/index_files/vgg_flower/"$A".index
done



mkdir "/data/gpfs/projects/punim1193/public_datasets/meta-dataset/index_files/quickdraw/"
for filename in /data/gpfs/projects/punim1193/public_datasets/meta-dataset/processed_data/quickdraw/*.tfrecords; do
    f="$echo${filename##*/}"
    A="$(cut -d'.' -f1 <<<"$f")"
    python3 -m tfrecord.tools.tfrecord2idx "$filename" /data/gpfs/projects/punim1193/public_datasets/meta-dataset/index_files/quickdraw/"$A".index
done


echo "Done"
