import json
import pickle
import os
import random
import string
import collections
import csv
from tqdm import tqdm
import re
import numpy as np
import spacy.cli
import spacy


def tqdm_wrap(iterable, *args, **kwargs):
    if len(iterable) > 100:
        return tqdm(iterable, *args, **kwargs)
    return iterable


class KvretData:
    def __init__(self, fileName):
        self.lines = {}
        self.conversations = []

        [self.lines, self.conversations] = self.loadLines(fileName)

    def loadLines(self, fileName):
        conversation = []
        lines = {}
        conversationId = 0
        lineID = 1
        print(fileName)
        with open(fileName, 'r') as f:
            datastore = json.load(f)
            for dialogue in datastore:
                convObj = {}
                conversationId = conversationId + 1
                convObj["lines"] = []
                for utterence in dialogue["dialogue"]:
                    lineID = lineID + 1
                    lineObj = {}
                    lineObj['turn'] = utterence['turn']
                    lineObj['utterance'] = utterence['data']['utterance']
                    if lineObj['turn'] == 'assistant':
                        requested = []
                        for knowledgeRequested in utterence['data']['requested']:
                            if utterence['data']['requested'][knowledgeRequested]:
                                requested.append(knowledgeRequested)
                        lineObj["requested"] = requested
                        lineObj["slots"] = utterence['data']['slots']
                    lines[lineID] = lineObj
                    convObj["lines"].append(lineObj)
                # EOS
                convObj[conversationId] = conversationId

                # Get KB entries
                predicate = []
                subject = None
                convObj["intent"] = dialogue["scenario"]["task"]["intent"]
                for col in dialogue["scenario"]["kb"]["column_names"]:
                    if subject is None:
                        subject = col
                        predicate.append(col)
                    else:
                        predicate.append(col)
                triples = []
                if dialogue["scenario"]["kb"]["items"] is not None:
                    for items in dialogue["scenario"]["kb"]["items"]:
                        for pred in predicate:
                            if (pred in items):
                                if items[pred] == items[subject]:
                                    triples.append([convObj["intent"], pred, items[pred]])
                                else:
                                    triples.append([items[subject], pred, items[pred]])
                            else:
                                triples.append([items[subject], pred, "-"])
                convObj["kb"] = triples
                # convObj["kb"] = []

                conversation.append(convObj)
        return [lines, conversation]

    def getConversations(self):
        return self.conversations


class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.encoderSeqs = []
        self.encoderSeqsLen = []
        self.decoderSeqs = []
        self.decoderSeqsLen = []
        self.seqIntent = []
        self.kb_inputs = []
        self.kb_inputs_mask = []
        self.targetKbMask = []
        self.targetSeqs = []
        self.weights = []
        self.encoderMaskSeqs = []
        self.decoderMaskSeqs = []
        self.triples_hist = []

class TextData:
    availableCorpus = collections.OrderedDict([  # OrderedDict because the first element is the default choice
        ('kvret', KvretData),
    ])

    @staticmethod
    def corpusChoices():
        """Return the dataset availables
        Return:
            list<string>: the supported corpus
        """
        return list(TextData.availableCorpus.keys())

    def __init__(self, dataFile, validFile, testFile, data_path):
        self.vocabularySize = 0
        self.corpus = 'kvret'
        self.data_path = data_path

        self.corpusDir = os.path.join(dataFile)
        self.validcorpus = os.path.join(validFile)
        self.testcorpus = os.path.join(testFile)

        self.fullSamplesPath = data_path + '/fullaSmples.pkl'
        self.filteredSamplesPath = data_path + '/filteredSamples.pkl'

        self.padToken = -1
        self.sosToken = -1
        self.eosToken = -1
        self.unknownToken = -1

        self.trainingSamples = []  # 2d array containing each question and its answer [[input,target,kb]]
        self.txtTrainingSamples = []
        self.txtValidationSamples = []
        self.validationSamples = []
        self.testSamples = []
        self.entities_property = dict()

        self.word2id = {}
        self.id2word = {}
        self.idCount = {}
        self.intent2id = {}
        self.id2intent = {}
        self.nlp = None
        self.loadCorpus()

        self.maxLengthEnco = self.getInputMaxLength()
        self.maxLengthDeco = self.getTargetMaxLength()
        self.maxTriples = self.getMaxTriples()
        self._printStats()

    def _printStats(self):
        print('Loaded Kvret : {} words, {} QA'.format(len(self.word2id), len(self.trainingSamples)))

    def shuffle(self):
        """Shuffle the training samples
        """
        random.shuffle(self.trainingSamples)

    def getMaxBatchSentLength(self, samples):
        max_q = np.max([len(sample[0]) for sample in samples])
        max_a = np.max([len(sample[1]) for sample in samples])
        max_kb = np.max([len(sample[2]) for sample in samples])
        return max_q, max_a, max_kb

    def createMyBatch(self, samples, transpose=True, additional_intent=False):
        batch = Batch()
        batchSize = len(samples)

        max_q, max_a, max_kb = self.getMaxBatchSentLength(samples)
        max_a = max_a + 2

        for i in range(batchSize):
            sample = samples[i]
            batch.encoderSeqs.append(sample[0])
            batch.decoderSeqs.append([self.sosToken] + sample[1] + [self.eosToken])
            batch.decoderMaskSeqs.append(list(np.ones(len(sample[1]) + 1)))
            batch.targetSeqs.append(
                batch.decoderSeqs[-1][1:])  # Same as decoder, but shifted to the left (ignore the <go>)
            batch.encoderMaskSeqs.append(list(np.ones(len(sample[0]))))
            batch.kb_inputs.append(sample[2])

            #batch.seqIntent.append(sample[3])
            batch.triples_hist.append(sample[3])
            batch.encoderSeqsLen.append(len(sample[0]))
            batch.decoderSeqsLen.append(len(sample[1]) + 1)

            batch.targetKbMask.append(list(np.ones(len(sample[2]))))

            if len(batch.encoderSeqs[i]) > self.maxLengthEnco:
                batch.encoderSeqs[i] = batch.encoderSeqs[i][self.maxLengthEnco:]
                batch.encoderMaskSeqs[i] = batch.encoderMaskSeqs[i][self.maxLengthEnco:]
            if len(batch.targetSeqs[i]) > self.maxLengthDeco:
                batch.decoderSeqs[i] = batch.decoderSeqs[i][:self.maxLengthDeco]
                batch.targetSeqs[i] = batch.targetSeqs[i][:self.maxLengthDeco]
                batch.decoderMaskSeqs[i] = batch.decoderMaskSeqs[i][:self.maxLengthDeco]
                batch.targetKbMask[i] = batch.targetKbMask[i][:self.maxLengthDeco]

            batch.encoderSeqs[i] = batch.encoderSeqs[i] + [self.padToken] * (max_q - len(batch.encoderSeqs[i]))
            # Left padding for the input
            batch.encoderMaskSeqs[i] = batch.encoderMaskSeqs[i] + [self.padToken] * (
                        max_q - len(batch.encoderMaskSeqs[i]))

            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.padToken] * (
                    max_a - len(batch.decoderSeqs[i]))
            batch.decoderMaskSeqs[i] = batch.decoderMaskSeqs[i] + [0] * (
                    max_a - len(batch.decoderMaskSeqs[i]))

            batch.targetKbMask[i] = batch.targetKbMask[i] + [self.padToken] * (
                    (max_kb) - len(batch.targetKbMask[i]))

            kb_pad_token = [0, 0, 0]
            batch.kb_inputs[i] = batch.kb_inputs[i] + [kb_pad_token] * (
                    (max_kb) - len(batch.kb_inputs[i]))
        return batch

    def getTestingBatch(self, batch_size=1):
        self.batchSize = batch_size
        batches = []

        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, self.getSampleSize(), self.batchSize):
                yield self.trainingSamples[i:min(i + self.batchSize, self.getSampleSize())]

        for samples in genNextSamples():
            batch = self.createMyBatch(samples, False)
            batches.append(batch)
            break
        return batches

    def get_kb_mask(self, sentence, kb):
        kb_mask = list(np.zeros(len(sentence)))
        for i, word in enumerate(sentence):
            for triple in kb:
                if triple[0] == word or triple[2] == word:
                    kb_mask[i] = 1
                    break
                else:
                    kb_mask[i] = 0

        assert len(kb_mask) == len(sentence)

        return kb_mask

    def getBatches(self, batch_size=1, valid=False, test=False, transpose=True):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        if (not valid or not test):
            self.shuffle()

        self.batchSize = batch_size

        batches = []

        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, len(self.trainingSamples), batch_size):
                if len(self.trainingSamples) > (i + batch_size):
                    yield self.trainingSamples[i:(i + batch_size)]
                else:
                    yield self.trainingSamples[-batch_size:]

        def genValidNextSamples():
            """ Generator over the mini-batch validation samples
                """
            for i in range(0, len(self.validationSamples), batch_size):
                if len(self.validationSamples) > (i + batch_size):
                    yield self.validationSamples[i:min(i + batch_size, len(self.validationSamples))]
                else:
                    yield self.validationSamples[-batch_size:]

        def genTestNextSamples():
            """ Generator over the mini-batch test samples
                """
            for i in range(0, len(self.testSamples), batch_size):
                if len(self.testSamples) > (i + batch_size):
                    yield self.testSamples[i:min(i + batch_size, len(self.testSamples))]
                else:
                    yield self.testSamples[-batch_size:]

        if valid:
            for samples in genValidNextSamples():
                batch = self.createMyBatch(samples, transpose)
                batches.append(batch)
        elif test:
            for samples in genTestNextSamples():
                batch = self.createMyBatch(samples, transpose)
                batches.append(batch)
        else:
            for samples in genNextSamples():
                batch = self.createMyBatch(samples, transpose)
                batches.append(batch)

        return batches

    def getSampleSize(self):
        return len(self.trainingSamples)

    def getVocabularySize(self):
        return len(self.word2id)

    def get_candidates(self, target_batches, all_predictions, references_list=False):
        candidate_sentences = []
        reference_sentences = []
        for target_batch, pridictions in zip(target_batches, all_predictions):
            for target, pridiction in zip(target_batch, pridictions):
                if references_list:
                    reference_sentences.append(self.sequence2str(target, clean=True))
                else:
                    reference_sentences.append([self.sequence2str(target, clean=True)])
                candidate_sentences.append(self.sequence2str(pridiction, clean=True, tensor=True))
        return candidate_sentences, reference_sentences

    def loadCorpus(self):
        # Try to construct the dataset from the preprocessed entry

        datasetExist = os.path.isfile(self.fullSamplesPath)
        if not datasetExist:
            print('Constructing full dataset...')
            self.nlp = spacy.load('en_core_web_sm')

            corpusData = TextData.availableCorpus['kvret'](self.corpusDir)
            validData = TextData.availableCorpus['kvret'](self.validcorpus)
            testData = TextData.availableCorpus['kvret'](self.testcorpus)

            self.createFullCorpus(corpusData.getConversations())
            self.createFullCorpus(validData.getConversations(), valid=True)
            self.createFullCorpus(testData.getConversations(), test=True)

            self.saveDataset(self.fullSamplesPath)
        else:
            self.loadDataset(self.fullSamplesPath)

        self._printStats()

        print('Filtering words (vocabSize = {} )...'.format(
            self.getVocabularySize()
        ))

        print('Saving dataset...')

    def saveDataset(self, filename):
        """Save samples to file
        Args:
            filename (str): pickle filename
        """

        with open(os.path.join(filename), 'wb') as handle:
            data = {
                'word2id': self.word2id,
                'id2word': self.id2word,
                'idCount': self.idCount,
                'intent2id': self.intent2id,
                'id2intent': self.id2intent,
                'trainingSamples': self.trainingSamples,
                'validationSamples': self.validationSamples,
                'testSamples': self.testSamples,
                'entities': self.entities_property,
            }
            pickle.dump(data, handle, -1)
            with open(self.data_path + "/train.csv", "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerows(self.txtTrainingSamples)
            with open(self.data_path + "/valid.csv", "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerows(self.txtValidationSamples)

    def loadDataset(self, filename):
        """Load samples from file
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2id = data['word2id']
            self.id2word = data['id2word']
            self.intent2id = data['intent2id']
            self.id2intent = data['id2intent']
            self.idCount = data.get('idCount', None)
            self.trainingSamples = data['trainingSamples']
            self.validationSamples = data['validationSamples']
            self.testSamples = data['testSamples']
            self.entities_property = data['entities']
            self.padToken = self.word2id['<pad>']
            self.sosToken = self.word2id['<sos>']
            self.eosToken = self.word2id['<eos>']
            self.hisToken = self.word2id['<his>']
            self.unknownToken = self.word2id['<unknown>']  # Restore special words

    def filterFromFull(self):
        """ Load the pre-processed full corpus and filter the vocabulary / sentences
        to match the given model options
        """

        def mergeSentences(sentences, fromEnd=False):
            """Merge the sentences until the max sentence length is reached
            Also decrement id count for unused sentences.
            Args:
                sentences (list<list<int>>): the list of sentences for the current line
                fromEnd (bool): Define the question on the answer
            Return:
                list<int>: the list of the word ids of the sentence
            """
            # We add sentence by sentence until we reach the maximum length
            merged = []

            # If question: we only keep the last sentences
            # If answer: we only keep the first sentences
            if fromEnd:
                sentences = reversed(sentences)

            for sentence in sentences:

                # If the total length is not too big, we still can add one more sentence
                if len(merged) + len(sentence) <= self.maxLength:
                    if fromEnd:  # Append the sentence
                        merged = sentence + merged
                    else:
                        merged = merged + sentence
                else:  # If the sentence is not used, neither are the words
                    for w in sentence:
                        self.idCount[w] -= 1
            return merged

        newSamples = []

        # 1st step: Iterate over all words and add filters the sentences
        # according to the sentence lengths
        for inputWords, targetWords, triples, intents in tqdm(self.trainingSamples, desc='Filter sentences:',
                                                              leave=False):
            # inputWords = mergeSentences(inputWords, fromEnd=True)
            # targetWords = mergeSentences(targetWords, fromEnd=False)

            newSamples.append([inputWords, targetWords, triples, intents])
        words = []

        # WARNING: DO NOT FILTER THE UNKNOWN TOKEN !!! Only word which has count==0 ?

        # 2nd step: filter the unused words and replace them by the unknown token
        # This is also where we update the correnspondance dictionaries
        specialTokens = {  # TODO: bad HACK to filter the special tokens. Error prone if one day add new special tokens
            self.padToken,
            self.sosToken,
            self.eosToken,
            self.unknownToken
        }

        newMapping = {}  # Map the full words ids to the new one (TODO: Should be a list)
        newId = 0

        selectedWordIds = collections \
            .Counter(self.idCount) \
            .most_common(self.vocabularySize or None)  # Keep all if vocabularySize == 0
        selectedWordIds = {k for k, v in selectedWordIds}  # if v > self.filterVocab}
        selectedWordIds |= specialTokens

        for wordId, count in [(i, self.idCount[i]) for i in range(len(self.idCount))]:  # Iterate in order
            if wordId in selectedWordIds:  # Update the word id
                newMapping[wordId] = newId
                word = self.id2word[wordId]  # The new id has changed, update the dictionaries
                del self.id2word[wordId]  # Will be recreated if newId == wordId
                self.word2id[word] = newId
                self.id2word[newId] = word
                newId += 1
            else:  # Cadidate to filtering, map it to unknownToken (Warning: don't filter special token)
                newMapping[wordId] = self.unknownToken
                del self.word2id[self.id2word[wordId]]  # The word isn't used anymore
                del self.id2word[wordId]

        # Last step: replace old ids by new ones and filters empty sentences
        def replace_words(words):
            valid = False  # Filter empty sequences
            for i, w in enumerate(words):
                words[i] = newMapping[w]
                if words[i] != self.unknownToken:  # Also filter if only contains unknown tokens
                    valid = True
            return valid

        for inputWords, targetWords, triples, intent in tqdm(newSamples, desc='Replace ids:', leave=False):
            valid = True
            valid &= replace_words(inputWords)
            valid &= replace_words(targetWords)
            valid &= targetWords.count(self.unknownToken) == 0  # Filter target with out-of-vocabulary target words ?

            if valid:
                self.trainingSamples.append(
                    [inputWords, targetWords, triples, intents])  # TODO: Could replace list by tuple

        self.idCount.clear()  # Not usefull anymore. Free data

    def createFullCorpus(self, conversations, valid=False, test=False):
        """Extract all data from the given vocabulary.
        Save the data on disk. Note that the entire corpus is pre-processed
        without restriction on the sentence length or vocab size.
        """
        # Add standard tokens
        self.padToken = self.getWordId('<pad>')
        self.sosToken = self.getWordId('<sos>')
        self.eosToken = self.getWordId('<eos>')
        self.hisToken = self.getWordId('<his>')
        self.unknownToken = self.getWordId('<unknown>')
        for conversation in tqdm(conversations, desc='Extract conversations'):
            self.extractConversation(conversation, valid, test)

        # The dataset will be saved in the same order it has been extracted

    def extractConversation(self, conversation, valid, test, herarical=False, truncate=False):
        """Extract the sample lines from the conversations
        Args:
            conversation (Obj): a conversation object containing the lines to extract
        """

        step = 2

        # Iterate over all the lines of the conversation
        input_conversation = []
        output_conversation = []
        input_txt_conversation = []
        output_txt_conversation = []
        triples = self.extractText(conversation['kb'], kb=True, train=not (valid or test))
        targetIntent = self.extractText(conversation['intent'], intent=True, train=not (valid or test))
        entity_tracker=[]
        for i in tqdm_wrap(
                range(0, len(conversation['lines']) - 1, step),  # We ignore the last line (no answer for it)
                desc='Conversation',
                leave=False):
            if herarical:
                if conversation['lines'][i]['turn'] == 'driver':
                    inputLine = conversation['lines'][i]
                    targetLine = conversation['lines'][i + 1]
                    input_conversation.extend(self.extractText(inputLine['utterance']))
                    output_conversation.extend(self.extractText(targetLine['utterance']))

                    if i < (len(conversation['lines']) - 2):
                        input_conversation.append('eou')
                        output_conversation.append('eou')

            else:

                if conversation['lines'][i]['turn'] == 'driver':
                    targeState = "Unknown"
                    inputLine = conversation['lines'][i]
                    targetLine = conversation['lines'][i + 1]
                    if "slots" in targetLine:
                        targeState = targetLine["slots"]

                    if i >= 1:
                        # input_conversation.append(self.eouToken)
                        input_conversation.extend(output_conversation)
                        if self.hisToken in input_conversation:
                            input_conversation.remove(self.hisToken)
                        input_conversation.append(self.hisToken)

                        # backup for text samples
                        # input_txt_conversation.append("<eou>")
                        input_txt_conversation.append(output_txt_conversation)
                        input_txt_conversation.append(self.id2word[self.hisToken])
                        # input_txt_conversation.append("<eou>")

                    input_txt_conversation.append(inputLine['utterance'])
                    output_txt_conversation = targetLine['utterance']

                    input_conversation.extend(
                        self.extractText(inputLine['utterance'], triples, train=not (valid or test)))
                    output_conversation = self.extractText(targetLine['utterance'], triples, train=not (valid or test))
                    out_with_intent = output_conversation
                    triples_hist = self.extractTriples(input_conversation,triples)

                if not valid and not test:  # Filter wrong samples (if one of the list is empty)
                    if truncate and (len(input_conversation[:]) >= 40 or len(output_conversation[:]) >= 40):
                        # truncate if too long
                        self.trainingSamples.append(
                            [input_conversation[len(input_conversation) - 40:], out_with_intent[:40], triples,
                             triples_hist])
                        self.txtTrainingSamples.append(
                            [np.array2string(np.array(input_txt_conversation[:]).flatten()).strip("]").strip("["),
                             self.sequence2str(out_with_intent[:]), triples_hist])
                    else:
                        self.trainingSamples.append([input_conversation[:], out_with_intent[:], triples, triples_hist])
                        self.txtTrainingSamples.append(
                            [self.sequence2str(input_conversation[:]), self.sequence2str(out_with_intent[:]),
                             self.sequence2str(triples_hist)])
                elif valid:
                    if truncate and (len(input_conversation[:]) >= 40 or len(output_conversation[:]) >= 40):
                        self.validationSamples.append(
                            [input_conversation[len(input_conversation) - 40:], output_conversation[:40], triples,
                             triples_hist])
                    else:
                        self.validationSamples.append(
                            [input_conversation[:], output_conversation[:], triples, triples_hist])
                    self.txtValidationSamples.append(
                        [self.sequence2str(input_conversation[:]), self.sequence2str(out_with_intent[:]),
                         self.sequence2str(triples_hist)])
                elif test:
                    self.testSamples.append([input_conversation[:], output_conversation[:], triples, triples_hist])

    def extractText(self, line, triples=[], kb=False, intent=False, train=True):
        """Extract the words from a sample lines
        Args:
            line (str): a line containing the text to extract
        Return:
            list<list<int>>: the list of sentences of word ids of the sentence
        """
        if intent:
            if line not in self.intent2id.keys():
                self.intent2id[line] = len(self.intent2id)
                self.id2intent[len(self.intent2id) - 1] = line

            return self.intent2id[line]

        if kb:
            triples = []
            entities_property = {}
            for triple in line:
                entities = []
                for entity in triple:
                    entity = entity.replace(".", "")
                    if len(re.split(',', entity.lower())) > 1:
                        for i, k in enumerate(re.split(',', entity.lower())):

                            processed_entity = "_".join(re.findall(r"[\w']+|[^\s\w']",
                                                                   " ".join(re.split('(\d+)(?=[a-z]|\-)',
                                                                                     k.strip().replace(".", "")))))
                            if len(entities) == 3:
                                triples.append(entities[:])
                                if not (entities[2] in self.entities_property.keys()):
                                    self.entities_property[entities[2]] = entities[1]
                                entities.pop()
                                entities.append(self.getWordId(processed_entity.lower(), train))

                            else:
                                entities.append(self.getWordId(processed_entity.lower(), train))

                    else:
                        processed_entity = "_".join(re.findall(r"[\w']+|[^\s\w']",
                                                               " ".join(re.split('(\d+)(?=[a-z]|\-)',
                                                                                 entity.strip().lower()))))
                        entities.append(self.getWordId(processed_entity.lower(), train))

                if not (entities[2] in self.entities_property.keys()):
                    self.entities_property[entities[2]] = entities[1]
                triples.append(entities)
            return triples

        else:
            line = line.replace('.', '').replace(',', '').replace(')', '').replace("(", '').replace('"', '').replace(
                '?', '') \
                .replace('>', '').replace("!", '').replace(':', '').replace(';', '').replace("' ", " ")
            doc = self.nlp(line)
            line_tokens = []
            for token in doc:
                line_tokens.append(token.text)
            line = " ".join(line_tokens).lower()

            for ent in doc.ents:
                temp = (ent.text.strip()).split(" ")
                if len(temp) > 1 and ((ent.label_ == 'TIME' and len(temp) < 3) or ent.label_ == 'GPE'):
                    line = line.replace(ent.text.lower(), '_'.join(temp).lower())
            count = 0
            entities = {}

            for ki in triples:
                ki_text = self.sequence2str(ki).split()

                object = " ".join(ki_text[2].split('_'))

                if object in line:
                    count = count + 1
                    line_temp = re.sub("(?![a-z]|[1-9])*" + object, " _entity_" + str(count) + "_", line)
                    line_temp = re.sub("_entity_[0-9]_[a-z|']{1,}", "_entity_" + str(count) + "_", line_temp)
                    if "_entity_" + str(count) + "_" in line_temp.split(" "):
                        line = line_temp
                        entities["_entity_" + str(count) + "_"] = ki_text[2]

                subject = " ".join(ki_text[0].split('_'))

                if subject in line:
                    count = count + 1
                    line_temp = re.sub("(?![a-z]|[1-9])*" + subject, "_entity_" + str(count) + "_", line)
                    line_temp = re.sub("_entity_[0-9]_[a-z|']{1,}", "_entity_" + str(count) + "_", line_temp)
                    if "_entity_" + str(count) + "_" in line_temp.split(" "):
                        line = line_temp
                        entities["_entity_" + str(count) + "_"] = ki_text[0]

            # Now to replace 50-60 by low and high degrees
            p = re.compile("\\b(\d{2} - \d{2,3}(f| degrees|s)+)\\b")
            x = p.findall(line)
            for degrees in x:
                low = "low_of_" + degrees[0].split("-")[0].strip() + "_f"
                high = "high_of_" + degrees[0].split("-")[1].strip()
                high = high.split(" ")[0] + '_f'
                line = re.sub(degrees[0], low + ' and ' + high, line)

            line = line.replace(" - ", " ")
            sentences = []  # List[List[str]]
            # Extract sentences
            sentencesToken = line.lower().split(" ")

            # We add sentence by sentence until we reach the maximum length
            for i in range(len(sentencesToken)):
                if sentencesToken[i] in entities:
                    token = entities[sentencesToken[i]]
                    sentences.append(self.getWordId(token, train))
                else:
                    token = sentencesToken[i]
                    if len(token) == 0:
                        continue
                    sentences.append(self.getWordId(token, train))  # Create the vocabulary and the training sentences

            return sentences

    def getWordId(self, word, create=True):
        """Get the id of the word (and add it to the dictionary if not existing). If the word does not exist and
        create is set to False, the function will return the unknownToken value
        Args:
            word (str): word to add
            create (Bool): if True and the word does not exist already, the world will be added
        Return:
            int: the id of the word created
        """
        # Should we Keep only words with more than one occurrence ?

        word = word.lower()
        if not create:
            wordId = self.word2id.get(word, self.unknownToken)
        elif word in self.word2id:
            wordId = self.word2id[word]
            self.idCount[wordId] += 1
        else:
            wordId = len(self.word2id)
            self.word2id[word] = wordId
            self.id2word[wordId] = word
            self.idCount[wordId] = 1

        return wordId

    def sequence2str(self, sequence, clean=False, reverse=False, tensor=False):
        """Convert a list of integer into a human readable string
        Args:
            sequence (list<int>): the sentence to print
            clean (Bool): if set, remove the <go>, <pad> and <eos> tokens
            reverse (Bool): for the input, option to restore the standard order
        Return:
            str: the sentence
        """
        try:
            if len(sequence) == 0:
                return ''
        except:
            print(sequence)
        if tensor:
            sequence = sequence.cpu().numpy()

        if not clean:
            return ' '.join([self.id2word[idx] for idx in sequence])

        sentence = []
        for wordId in sequence:
            if wordId == self.eosToken:  # End of generated sentence
                sentence.append(self.id2word[wordId])
                break
            elif wordId != self.padToken and wordId != self.sosToken:
                sentence.append(self.id2word[wordId])

        if reverse:  # Reverse means input so no <eos> (otherwise pb with previous early stop)
            sentence.reverse()

        return self.detokenize(sentence)

    def detokenize(self, tokens):
        """Slightly cleaner version of joining with spaces.
        Args:
            tokens (list<string>): the sentence to print
        Return:
            str: the sentence
        """
        return ''.join([
            ' ' + t if not t.startswith('\'') and
                       t not in string.punctuation
            else t
            for t in tokens]).strip().capitalize()

    def getInputMaxLength(self):
        maxT = max(map(len, (s for [s, _, _, _] in self.trainingSamples)))
        return maxT

    def getTargetMaxLength(self):
        maxT = max(map(len, (s for [_, s, _, _] in self.trainingSamples)))
        return maxT + 2

    def getMaxTriples(self):
        return max(map(len, (s for [_, _, s, _] in self.trainingSamples)))

    def extractTriples(self, input_conversation, triples):
        if len(triples)==0:
            return []
        existing_tripes = []
        flatten_entities = np.array(triples).flatten()
        for word in input_conversation:
            if word in flatten_entities and not word in existing_tripes:
                existing_tripes.append(word)
        return existing_tripes


