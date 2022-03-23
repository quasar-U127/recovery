PROJECT_PREFIX=$SCRATCH

export STGAN_DATA=$PROJECT_PREFIX/STGAN
data-manager --root $PROJECT_PREFIX clone ada:$ADA_HDD/projects/STGAN 
data-manager --root $STGAN_DATA pull output/128/undo_features

export PROJECT_DATA=$PROJECT_PREFIX/recovery
data-manager --root $PROJECT_PREFIX clone ada:$ADA_HDD/projects/recovery 
data-manager --root $PROJECT_DATA pull .

export SMALL_CELEBA=$STGAN_DATA/small_celeba


