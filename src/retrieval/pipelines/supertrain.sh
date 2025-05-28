echo Preprocessing data. 
bash src/retrieval/pipelines/preprocess_msmarco.sh

echo Training MS Marco Rocket for three passage epochs and one document epoch. 
bash src/retrieval/pipelines/train_rocket_lt.sh

echo Training MS Marco Rocket with negatives. 
bash src/retrieval/pipelines/negative_rocket_lt.sh

echo Training MS Marco Gradient Rocket. 
bash src/retrieval/pipelines/gradient_train_rocket_lt.sh

echo Training MS Marco Rocket with negatives and triplet loss. 
bash src/retrieval/pipelines/negative_rocket_lt_tripletloss.sh