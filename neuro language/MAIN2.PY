import cupy as cp
import numpy as np
import string
import requests
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from transformers import pipeline
from scipy import sparse
nltk.download('punkt')

# 뉴런 및 시냅스 수 설정 (인간 뇌 비율에 맞춤)
TOTAL_NEURONS = 860000  # 인간 뇌 1/1,000,000 수준
LANGUAGE_NEURONS = int(TOTAL_NEURONS * 0.186)  # 대뇌 피질: 160,000
CONTEXT_NEURONS = int(TOTAL_NEURONS * 0.602)  # 소뇌: 518,000
EMOTION_NEURONS = int(TOTAL_NEURONS * 0.012)  # 편도체: 10,000
THALAMUS_NEURONS = int(TOTAL_NEURONS * 0.1)   # 시상: 86,000
HIPPOCAMPUS_NEURONS = int(TOTAL_NEURONS * 0.1)  # 해마: 86,000
PROCESSING_NEURONS = LANGUAGE_NEURONS  # 처리 레이어
AVG_SYNAPSES_PER_NEURON = 7000  # 인간 뇌 평균
SPARSITY = 0.008  # 희소 연결 밀도

# 오타 교정
def correct_typo(word, candidates):
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    distances = [(candidate, levenshtein_distance(word, candidate)) for candidate in candidates]
    return min(distances, key=lambda x: x[1])[0]

# 감정 데이터 로드
def load_emotion_data():
    texts = ["I am so happy today", "I feel very sad", "This makes me angry", "I'm scared of this", "It's just okay"]
    labels = ["joy", "sadness", "anger", "fear", "neutral"]
    return texts, labels

# 단어 사전 관리
class Vocabulary:
    def __init__(self):
        self.word2idx = defaultdict(lambda: len(self.word2idx))
        self.idx2word = {}
        self.word2idx['<PAD>']
        self.word2idx['<UNK>']

    def build(self, texts):
        for text in texts:
            for sentence in sent_tokenize(text):
                for word in word_tokenize(sentence.lower()):
                    idx = self.word2idx[word]
                    self.idx2word[idx] = word
        return self

    def encode(self, words):
        return [self.word2idx.get(word.lower(), self.word2idx['<UNK>']) for word in words]

    def decode(self, indices):
        return [self.idx2word.get(idx, '<UNK>') for idx in indices]

# 희소 가중치 행렬 생성
def create_sparse_weights(pre_size, post_size, density=SPARSITY):
    weights = sparse.random(pre_size, post_size, density=density, data_rvs=lambda size: np.random.uniform(0.1, 1.0, size))
    return cp.array(weights.toarray())

class NeuronLayer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.potentials = cp.full((num_neurons,), -70.0)
        self.thresholds = cp.full((num_neurons,), -55.0)
        self.absolute_refractory_period = cp.zeros(num_neurons)
        self.memory = cp.zeros(num_neurons)
        self.na_in = cp.full((num_neurons,), 15.0)
        self.na_out = cp.full((num_neurons,), 145.0)
        self.k_in = cp.full((num_neurons,), 140.0)
        self.k_out = cp.full((num_neurons,), 4.0)
        self.ca_in = cp.full((num_neurons,), 0.0001)
        self.ca_out = cp.full((num_neurons,), 2.0)
        self.cl_in = cp.full((num_neurons,), 10.0)
        self.cl_out = cp.full((num_neurons,), 110.0)
        self.h_concentration = cp.full((num_neurons,), 7.2)
        self.na_channel_open = cp.zeros(num_neurons, dtype=bool)
        self.k_channel_open = cp.zeros(num_neurons, dtype=bool)
        self.ca_channel_open = cp.zeros(num_neurons, dtype=bool)
        self.cl_channel_open = cp.zeros(num_neurons, dtype=bool)
        self.h_channel_open = cp.zeros(num_neurons, dtype=bool)
        # 억제 뉴런 추가
        self.inhibitory_neurons = cp.zeros(num_neurons, dtype=bool)
        self.inhibitory_weights = create_sparse_weights(num_neurons, num_neurons, density=0.01)

    def apply_inhibition(self):
        inhibitory_effect = cp.dot(self.potentials, self.inhibitory_weights)
        self.potentials -= inhibitory_effect
        self.potentials = cp.clip(self.potentials, -80.0, 50.0)

    def update_potential(self, time_step=0.1):
        self.potentials[self.na_channel_open] += 10.0 * time_step
        self.potentials[self.k_channel_open] -= 10.0 * time_step
        self.potentials[self.ca_channel_open] += 2.0 * time_step
        self.potentials[self.cl_channel_open] -= 5.0 * time_step
        self.potentials[self.h_channel_open] += 0.5 * time_step
        self.potentials += 0.1 * self.memory
        self.apply_inhibition()
        self.potentials = cp.clip(self.potentials, -80.0, 50.0)

    def update_ion_pumps(self):
        pump_activity = 0.1
        self.na_in += pump_activity * (self.na_out - self.na_in) * 0.1
        self.k_in -= pump_activity * (self.k_out - self.k_in) * 0.1
        self.ca_in -= 0.05 * self.ca_in
        self.h_concentration -= 0.02 * (self.h_concentration - 7.2)

    def fire(self):
        fired = self.potentials >= self.thresholds
        self.potentials[fired] = -70.0
        self.memory[fired] = self.potentials[fired]
        return fired

    def update(self, time_step=0.1):
        self.update_potential(time_step)
        self.update_ion_pumps()
        self.memory *= 0.9

class ThalamusNeuronLayer(NeuronLayer):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)
        self.relay_weights = create_sparse_weights(num_neurons, num_neurons)

    def relay_information(self, input_potentials):
        relayed = cp.dot(input_potentials, self.relay_weights)
        self.potentials += relayed
        self.potentials = cp.clip(self.potentials, -80.0, 50.0)

class HippocampusNeuronLayer(NeuronLayer):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)
        self.long_term_memory = cp.zeros(num_neurons)
        self.memory_weights = create_sparse_weights(num_neurons, num_neurons)

    def store_long_term_memory(self, input_potentials):
        self.long_term_memory += cp.dot(input_potentials, self.memory_weights)
        self.long_term_memory = cp.clip(self.long_term_memory, -50.0, 50.0)

    def retrieve_memory(self):
        return self.long_term_memory

class MultiHeadAttentionNeuronLayer(NeuronLayer):
    def __init__(self, num_neurons, num_heads=4):
        super().__init__(num_neurons)
        self.num_heads = num_heads
        self.head_dim = num_neurons // num_heads
        self.query_weights = create_sparse_weights(num_neurons, self.head_dim * num_heads)
        self.key_weights = create_sparse_weights(num_neurons, self.head_dim * num_heads)
        self.value_weights = create_sparse_weights(num_neurons, self.head_dim * num_heads)

    def multi_head_attention(self, input_potentials):
        attended = cp.zeros(self.num_neurons)
        for head in range(self.num_heads):
            start = head * self.head_dim
            end = (head + 1) * self.head_dim
            queries = cp.dot(input_potentials, self.query_weights[:, start:end])
            keys = cp.dot(input_potentials, self.key_weights[:, start:end])
            values = cp.dot(input_potentials, self.value_weights[:, start:end])
            scores = cp.dot(queries, keys.T) / cp.sqrt(self.head_dim)
            attention_weights = cp.softmax(scores, axis=1)
            head_output = cp.dot(attention_weights, values)
            attended[start:end] += head_output
        self.potentials += attended
        self.potentials = cp.clip(self.potentials, -80.0, 50.0)

class MemoryNeuronLayer(NeuronLayer):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)
        self.memory_weights = create_sparse_weights(num_neurons, num_neurons)

    def update_memory(self, input_potentials):
        self.memory += cp.dot(input_potentials, self.memory_weights)
        self.memory = cp.clip(self.memory, -50.0, 50.0)

    def get_context(self):
        return self.memory

class EmotionNeuronLayer(NeuronLayer):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)
        self.emotion_types = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'neutral']
        self.emotion_states = cp.zeros(num_neurons)
        self.emotion_intensity = cp.zeros(num_neurons)
        self.emotion_weights = create_sparse_weights(num_neurons, len(self.emotion_types), density=0.1)

    def train_emotion(self, texts, labels):
        for text, label in zip(texts, labels):
            tokens = word_tokenize(text.lower())
            indices = [hash(word) % self.num_neurons for word in tokens]
            label_idx = self.emotion_types.index(label) if label in self.emotion_types else len(self.emotion_types) - 1
            for idx in indices:
                self.emotion_weights[idx, label_idx] += 0.1
        self.emotion_weights = cp.clip(self.emotion_weights, -5.0, 5.0)

    def update_emotion(self, input_potentials):
        emotion_scores = cp.dot(input_potentials, self.emotion_weights)
        self.emotion_states = cp.argmax(emotion_scores, axis=1)
        self.emotion_intensity = cp.max(emotion_scores, axis=1) / 10.0

    def get_emotion_influence(self):
        influence = cp.zeros_like(self.emotion_states, dtype=float)
        for i, emotion in enumerate(self.emotion_types):
            influence[self.emotion_states == i] = {
                'joy': 5.0, 'sadness': -5.0, 'anger': -3.0, 'fear': -2.0,
                'surprise': 2.0, 'disgust': -4.0, 'trust': 3.0, 'neutral': 0.0
            }[emotion] * self.emotion_intensity[self.emotion_states == i]
        return influence

class Synapse:
    def __init__(self, pre_layer, post_layer, bidirectional=False):
        self.pre_layer = pre_layer
        self.post_layer = post_layer
        self.weights_forward = create_sparse_weights(pre_layer.num_neurons, post_layer.num_neurons)
        self.weights_backward = create_sparse_weights(post_layer.num_neurons, pre_layer.num_neurons) if bidirectional else None
        self.timing = cp.zeros(pre_layer.num_neurons)

    def propagate_forward(self):
        signals = cp.dot(self.pre_layer.potentials, self.weights_forward)
        self.post_layer.potentials += signals

    def propagate_backward(self):
        if self.weights_backward is not None:
            signals = cp.dot(self.post_layer.potentials, self.weights_backward)
            self.pre_layer.potentials += signals

    def update_stdp(self, learning_rate=0.01):
        pre_fired = self.pre_layer.fire()
        post_fired = self.post_layer.fire()
        time_diff = self.timing[pre_fired] - self.timing[post_fired]
        delta_w = learning_rate * cp.exp(-cp.abs(time_diff) / 20.0)
        delta_w[time_diff > 0] *= 1.0
        delta_w[time_diff < 0] *= -0.5
        self.weights_forward[pre_fired, post_fired] += delta_w
        if self.weights_backward is not None:
            self.weights_backward[post_fired, pre_fired] += delta_w
        self.weights_forward = cp.clip(self.weights_forward, 0.0, 10.0)
        if self.weights_backward is not None:
            self.weights_backward = cp.clip(self.weights_backward, 0.0, 10.0)

class TokenLayer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.num_tokens = min(len(self.vocab.word2idx), LANGUAGE_NEURONS)
        self.neurons = NeuronLayer(self.num_tokens)

    def tokenize(self, text):
        sentences = sent_tokenize(text)
        tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
        return tokenized_sentences

    def encode(self, tokenized_sentences):
        return [self.vocab.encode(sentence)[:self.num_tokens] for sentence in tokenized_sentences]

    def decode(self, indices):
        return [self.vocab.decode(sentence[:self.num_tokens]) for sentence in indices]

class NLPModel:
    def __init__(self, num_layers=6):
        self.vocab = Vocabulary()
        self.token_layer = TokenLayer(self.vocab)
        self.emotion_layer = EmotionNeuronLayer(EMOTION_NEURONS)
        self.thalamus_layer = ThalamusNeuronLayer(THALAMUS_NEURONS)
        self.hippocampus_layer = HippocampusNeuronLayer(HIPPOCAMPUS_NEURONS)
        self.attention_layer = MultiHeadAttentionNeuronLayer(CONTEXT_NEURONS)
        self.memory_layer = MemoryNeuronLayer(CONTEXT_NEURONS)
        self.processing_layers = [NeuronLayer(PROCESSING_NEURONS) for _ in range(num_layers)]
        self.output_layer = NeuronLayer(PROCESSING_NEURONS)
        self.conversation_history = []
        self.current_topic = None
        self.text_generator = pipeline("text-generation", model="distilgpt2")
        self.text_corrector = pipeline("fill-mask", model="bert-base-uncased")
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

        # 복잡한 연결 패턴
        self.synapses = [
            Synapse(self.token_layer.neurons, self.thalamus_layer, bidirectional=True),  # 피질-시상 루프
            Synapse(self.thalamus_layer, self.attention_layer, bidirectional=True),  # 시상-소뇌
            Synapse(self.attention_layer, self.memory_layer),
            Synapse(self.memory_layer, self.processing_layers[0], bidirectional=True),  # 소뇌-피질 피드백
            Synapse(self.token_layer.neurons, self.emotion_layer, bidirectional=True),  # 피질-편도체 상호작용
            Synapse(self.token_layer.neurons, self.hippocampus_layer, bidirectional=True),  # 피질-해마 상호작용
            Synapse(self.emotion_layer, self.memory_layer, bidirectional=True),  # 감정-문맥 상호작용
        ]
        for i in range(num_layers - 1):
            self.synapses.append(Synapse(self.processing_layers[i], self.processing_layers[i + 1]))
        self.synapses.append(Synapse(self.processing_layers[-1], self.output_layer))
        self.feedback_synapse = Synapse(self.output_layer, self.token_layer.neurons, bidirectional=True)  # 출력-입력 피드백

    def train_emotion(self):
        texts, labels = load_emotion_data()
        self.emotion_layer.train_emotion(texts, labels)
        print("감정 학습 완료!")

    def train(self, texts, epochs=2000, batch_size=32):
        self.vocab.build(texts)
        self.token_layer = TokenLayer(self.vocab)
        training_sentences = [self.token_layer.tokenize(text) for text in texts]
        training_indices = [self.token_layer.encode(sentences) for sentences in training_sentences]

        max_length = min(max(max(len(sentence) for sentence in indices) for indices in training_indices), self.token_layer.num_tokens)
        padded_training_indices = []
        for indices in training_indices:
            padded_sentences = []
            for sentence in indices:
                padded_sentence = sentence[:max_length] + [self.vocab.word2idx['<PAD>']] * (max_length - len(sentence))
                padded_sentences.append(padded_sentence)
            padded_training_indices.append(padded_sentences)
        padded_training_indices = cp.array(padded_training_indices[:, :, :self.token_layer.num_tokens], dtype=cp.int32)

        for epoch in range(epochs):
            batch_losses = []
            batch_indices = cp.random.choice(len(padded_training_indices), size=batch_size, replace=False)
            batch_sentences = padded_training_indices[batch_indices]

            for batch in batch_sentences:
                for sentence in batch:
                    for i in range(len(sentence) - 1):
                        current_idx = int(sentence[i])
                        next_idx = int(sentence[i + 1])

                        self.token_layer.neurons.potentials[current_idx] = 50.0
                        self.synapses[0].propagate_forward()  # Token → Thalamus
                        self.thalamus_layer.relay_information(self.token_layer.neurons.potentials)
                        self.synapses[0].propagate_backward()  # Thalamus → Token

                        self.synapses[4].propagate_forward()  # Token → Emotion
                        self.emotion_layer.update_emotion(self.token_layer.neurons.potentials[:EMOTION_NEURONS])
                        emotion_influence = self.emotion_layer.get_emotion_influence()
                        self.token_layer.neurons.potentials[:EMOTION_NEURONS] += emotion_influence
                        self.synapses[4].propagate_backward()  # Emotion → Token

                        self.synapses[5].propagate_forward()  # Token → Hippocampus
                        self.hippocampus_layer.store_long_term_memory(self.token_layer.neurons.potentials)
                        hippocampus_memory = self.hippocampus_layer.retrieve_memory()
                        self.token_layer.neurons.potentials += 0.1 * hippocampus_memory[:self.token_layer.num_tokens]
                        self.synapses[5].propagate_backward()  # Hippocampus → Token

                        self.synapses[1].propagate_forward()  # Thalamus → Attention
                        self.attention_layer.multi_head_attention(self.thalamus_layer.potentials)
                        self.synapses[1].propagate_backward()  # Attention → Thalamus

                        self.synapses[2].propagate_forward()  # Attention → Memory
                        self.memory_layer.update_memory(self.attention_layer.potentials)

                        self.synapses[6].propagate_forward()  # Emotion → Memory
                        self.memory_layer.potentials += 0.2 * emotion_influence[:self.memory_layer.num_neurons]
                        self.synapses[6].propagate_backward()  # Memory → Emotion

                        self.synapses[3].propagate_forward()  # Memory → Processing[0]
                        for j in range(len(self.processing_layers) - 1):
                            self.synapses[7 + j].propagate_forward()
                        self.synapses[-1].propagate_forward()  # Processing[-1] → Output

                        self.feedback_synapse.propagate_forward()  # Output → Token (피드백)
                        self.feedback_synapse.propagate_backward()  # Token → Output

                        fired = self.output_layer.fire()
                        if next_idx < self.output_layer.num_neurons and fired[next_idx]:
                            for synapse in self.synapses:
                                synapse.weights_forward[current_idx, next_idx] += 0.005
                            batch_losses.append(1)

                        self.token_layer.neurons.update()
                        self.emotion_layer.update()
                        self.thalamus_layer.update()
                        self.hippocampus_layer.update()
                        self.attention_layer.update()
                        self.memory_layer.update()
                        for layer in self.processing_layers:
                            layer.update()
                        self.output_layer.update()

            if epoch % 100 == 0:
                avg_loss = sum(batch_losses) / max(len(batch_losses), 1)
                print(f"Epoch {epoch+1}/{epochs} - 평균 학습 정확도: {avg_loss:.4f}")

    def reinforce_train(self, conversations, epochs=100):
        for epoch in range(epochs):
            total_reward = 0
            for user_input, expected_response in conversations:
                predicted_response = self.predict(user_input, return_text=True)
                sentiment = self.sentiment_analyzer(predicted_response)[0]
                grok2_feedback = 1.0 if sentiment['label'] == 'POSITIVE' else -1.0
                grok2_naturalness = 0.5 if len(predicted_response.split()) > 3 else -0.5
                reward = grok2_feedback + grok2_naturalness
                total_reward += reward

                for synapse in self.synapses:
                    synapse.weights_forward += reward * 0.01
                    if synapse.weights_backward is not None:
                        synapse.weights_backward += reward * 0.01
                    synapse.weights_forward = cp.clip(synapse.weights_forward, 0.0, 10.0)
                    if synapse.weights_backward is not None:
                        synapse.weights_backward = cp.clip(synapse.weights_backward, 0.0, 10.0)

                self.feedback_synapse.weights_forward += reward * 0.01
                self.feedback_synapse.weights_backward += reward * 0.01
                self.feedback_synapse.weights_forward = cp.clip(self.feedback_synapse.weights_forward, 0.0, 10.0)
                self.feedback_synapse.weights_backward = cp.clip(self.feedback_synapse.weights_backward, 0.0, 10.0)

            print(f"강화 학습 Epoch {epoch+1}/{epochs} - 평균 보상: {total_reward/len(conversations):.4f}")

    def predict(self, text, length=5, return_text=False):
        self.conversation_history.append(text)
        tokenized_sentences = self.token_layer.tokenize(text)

        # 오타 교정
        candidates = list(self.vocab.word2idx.keys())
        corrected_sentences = []
        for sentence in tokenized_sentences:
            corrected_sentence = [correct_typo(word, candidates) if word not in candidates else word for word in sentence]
            corrected_sentences.append(corrected_sentence)
        tokenized_sentences = corrected_sentences

        # 대화 주제 추적
        topics = ["weather", "joke", "emotion"]
        for topic in topics:
            if topic in text.lower():
                self.current_topic = topic
                break

        indices = self.token_layer.encode(tokenized_sentences)
        output_indices = []

        for sentence in indices[-1:]:
            current_sentence = sentence.copy()[:self.token_layer.num_tokens]
            for _ in range(length):
                current_idx = current_sentence[-1]
                self.token_layer.neurons.potentials[current_idx] = 50.0
                self.synapses[0].propagate_forward()  # Token → Thalamus
                self.thalamus_layer.relay_information(self.token_layer.neurons.potentials)
                self.synapses[0].propagate_backward()  # Thalamus → Token

                self.synapses[4].propagate_forward()  # Token → Emotion
                self.emotion_layer.update_emotion(self.token_layer.neurons.potentials[:EMOTION_NEURONS])
                emotion_influence = self.emotion_layer.get_emotion_influence()
                self.token_layer.neurons.potentials[:EMOTION_NEURONS] += emotion_influence
                self.synapses[4].propagate_backward()  # Emotion → Token

                self.synapses[5].propagate_forward()  # Token → Hippocampus
                self.hippocampus_layer.store_long_term_memory(self.token_layer.neurons.potentials)
                hippocampus_memory = self.hippocampus_layer.retrieve_memory()
                self.token_layer.neurons.potentials += 0.1 * hippocampus_memory[:self.token_layer.num_tokens]
                self.synapses[5].propagate_backward()  # Hippocampus → Token

                self.synapses[1].propagate_forward()  # Thalamus → Attention
                self.attention_layer.multi_head_attention(self.thalamus_layer.potentials)
                self.synapses[1].propagate_backward()  # Attention → Thalamus

                self.synapses[2].propagate_forward()  # Attention → Memory
                self.memory_layer.update_memory(self.attention_layer.potentials)

                self.synapses[6].propagate_forward()  # Emotion → Memory
                self.memory_layer.potentials += 0.2 * emotion_influence[:self.memory_layer.num_neurons]
                self.synapses[6].propagate_backward()  # Memory → Emotion

                self.synapses[3].propagate_forward()  # Memory → Processing[0]
                for j in range(len(self.processing_layers) - 1):
                    self.synapses[7 + j].propagate_forward()
                self.synapses[-1].propagate_forward()  # Processing[-1] → Output

                self.feedback_synapse.propagate_forward()  # Output → Token (피드백)
                self.feedback_synapse.propagate_backward()  # Token → Output

                next_idx = int(cp.argmax(self.output_layer.potentials[:self.output_layer.num_neurons]))
                current_sentence.append(next_idx)

                self.token_layer.neurons.update()
                self.emotion_layer.update()
                self.thalamus_layer.update()
                self.hippocampus_layer.update()
                self.attention_layer.update()
                self.memory_layer.update()
                for layer in self.processing_layers:
                    layer.update()
                self.output_layer.update()

            output_indices.append(current_sentence)

        predicted_text = ' '.join(self.token_layer.decode(output_indices)[0])
        generated = self.text_generator(predicted_text, max_length=50, num_return_sequences=1)[0]['generated_text']
        corrected = generated

        # 감정 기반 어조 조정
        sentiment = self.sentiment_analyzer(corrected)[0]
        if sentiment['label'] == 'POSITIVE':
            corrected = corrected + " 😊"
        elif sentiment['label'] == 'NEGATIVE':
            corrected = corrected + " 😔"

        return corrected if not return_text else corrected

    def chat(self):
        print("안녕! 나와 대화하자. 언제든 '종료'라고 입력하면 대화를 끝낼게.")
        while True:
            user_input = input("너: ")
            if user_input.lower() == "종료":
                print("나: 대화를 끝내자. 안녕!")
                break
            response = self.predict(user_input, length=5)
            print(f"나: {response}")

def load_large_text_data():
    urls = ["https://www.gutenberg.org/files/1342/1342-0.txt"]
    texts = []
    for url in urls:
        response = requests.get(url)
        texts.append(response.text)
    return texts

def main():
    model = NLPModel(num_layers=6)
    training_texts = load_large_text_data()
    print(f"훈련 시작: {len(training_texts)}개의 문장 학습")
    model.train_emotion()
    model.train(training_texts, epochs=2000, batch_size=32)

    conversations = [
        ("Hello how are you", "I'm doing great, thanks for asking!"),
        ("What is the weather like", "It's sunny today, perfect for a walk."),
        ("Tell me a joke", "Why did the scarecrow become a motivational speaker? Because he was outstanding in his field!")
    ]
    model.reinforce_train(conversations, epochs=100)

    # 대화 시작
    model.chat()

if __name__ == "__main__":
    main()