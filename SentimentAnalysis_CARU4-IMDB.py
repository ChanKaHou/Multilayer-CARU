﻿'''
Python program source code
for research article "Multilayer CARU Networks for Data Stream Classification"
Version 1.0
(c) Copyright 2021 Ka-Hou Chan <chankahou (at) ipm.edu.mo>
The python program source code is free software: you can redistribute
it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.
The python program source code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.
You should have received a copy of the GNU General Public License
along with the Kon package.  If not, see <http://www.gnu.org/licenses/>.
'''

import os
import torch
import torchtext
import pytorch_lightning

class CARUCell(torch.nn.Module): #Content-Adaptive Recurrent Unit
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.LW = torch.nn.Linear(hidden_size, hidden_size)
        self.LL = torch.nn.Linear(input_size, hidden_size)
        self.Weight = torch.nn.Linear(hidden_size, hidden_size)
        self.Linear = torch.nn.Linear(input_size, hidden_size)

    def forward(self, word, hidden):
        feature = self.Linear(word)
        if hidden is None:
            return torch.tanh(feature)
        n = torch.tanh(self.Weight(hidden) + feature)
        l = torch.sigmoid(feature)*torch.sigmoid(self.LW(hidden) + self.LL(word))
        return torch.lerp(hidden, n, l)

class CARU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()

        self.CARUCells = torch.nn.ModuleList([CARUCell(hidden_size if (index>0) else input_size, hidden_size) for index in range(num_layers)])

    def forward(self, sequence):
        hidden = [None]*len(sequence)
        for CARUCell in self.CARUCells:
            for index, feature in enumerate(sequence):
                hidden[index] = CARUCell(feature, hidden[index-1] if (index>0) else None)
            sequence = hidden
        return sequence

class Model(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()

        #python -m spacy download en_core_web_sm
        self.Text = torchtext.legacy.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', lower=True)
        self.Label = torchtext.legacy.data.LabelField(is_target=True)

        self.trainSet, self.testSet = torchtext.legacy.datasets.IMDB.splits(self.Text, self.Label, root='../../data')
        print(len(self.trainSet), len(self.testSet)) #25000 25000

        self.Text.build_vocab(self.trainSet.text, vectors_cache='../../data/vector_cache', min_freq=4, vectors='glove.6B.100d')
        self.Label.build_vocab(self.trainSet.label)
        print('Text Vocabulary Size:', len(self.Text.vocab))
        print('Label Vocabulary Size:', len(self.Label.vocab))

        self.Embedding = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(self.Text.vocab.vectors),
            torch.nn.Dropout(),
            )
        self.CARU = CARU(100, 256, num_layers=4)
        self.Classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, len(self.Label.vocab)),
            )

    def prepare_data(self):
        return

    def setup(self, stage):
        if (stage == 'fit'):
            self.trainData, self.valData = self.trainSet.split([20000, 5000])
        elif (stage == 'test'):
            self.testData = self.testSet

    def forward(self, sentence):
        embedded = self.Embedding(sentence) #[S, batch_size, E]
        hidden = self.CARU(embedded)
        return self.Classifier(hidden[-1])
        
    def train_dataloader(self):
        return torchtext.legacy.data.BucketIterator(self.trainData, batch_size=100, shuffle=True)

    def val_dataloader(self):
        return torchtext.legacy.data.BucketIterator(self.valData, batch_size=100)

    def test_dataloader(self):
        return torchtext.legacy.data.BucketIterator(self.testData, batch_size=100)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'reduce_on_plateau': True, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        pred_label = self(batch.text)
        return torch.nn.functional.cross_entropy(pred_label, batch.label)

    def validation_step(self, batch, batch_idx):
        pred_label = self(batch.text)
        loss = torch.nn.functional.cross_entropy(pred_label, batch.label)
        acc = torch.mean(pred_label.argmax(-1) == batch.label, dtype=torch.float)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pred_label = self(batch.text)
        acc = torch.mean(pred_label.argmax(-1) == batch.label, dtype=torch.float)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        print()

model = Model()
trainer = pytorch_lightning.Trainer(gpus=-1, max_epochs=150)
trainer.fit(model)
trainer.test(model)
