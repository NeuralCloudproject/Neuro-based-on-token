import cupy as cp
import string
import requests

import cupy as cp

class Neuron:
    """뉴런 클래스: 막전위, 이온 통로, 이온 펌프 포함"""

    def __init__(self, num_neurons):
        self.num_neurons = num_neurons

        # 뉴런 상태 변수
        self.potentials = cp.full((num_neurons,), -70.0)  # 막전위 (mV)
        self.thresholds = cp.full((num_neurons,), -55.0)  # 발화 임계값 (mV)
        self.absolute_refractory_period = cp.zeros(num_neurons)  # 절대 불응기 (ms)

        # 🔹 이온 채널 상태 (Na⁺, K⁺, Ca²⁺, Cl⁻, H⁺)
        self.na_channel_open = cp.zeros(num_neurons, dtype=bool)  # Na⁺ 채널
        self.k_channel_open = cp.zeros(num_neurons, dtype=bool)   # K⁺ 채널
        self.ca_channel_open = cp.zeros(num_neurons, dtype=bool)  # Ca²⁺ 채널
        self.cl_channel_open = cp.zeros(num_neurons, dtype=bool)  # Cl⁻ 채널
        self.h_channel_open = cp.zeros(num_neurons, dtype=bool)   # H⁺ 채널 (pH 조절)

        # 🔹 이온 농도 (내부 & 외부)
        self.na_in = cp.full((num_neurons,), 15.0)    # 내부 Na⁺ (mM)
        self.na_out = cp.full((num_neurons,), 145.0)  # 외부 Na⁺ (mM)
        self.k_in = cp.full((num_neurons,), 140.0)    # 내부 K⁺ (mM)
        self.k_out = cp.full((num_neurons,), 4.0)     # 외부 K⁺ (mM)
        self.ca_in = cp.full((num_neurons,), 0.0001)  # 내부 Ca²⁺ (mM)
        self.ca_out = cp.full((num_neurons,), 2.0)    # 외부 Ca²⁺ (mM)
        self.cl_in = cp.full((num_neurons,), 10.0)    # 내부 Cl⁻ (mM)
        self.cl_out = cp.full((num_neurons,), 110.0)  # 외부 Cl⁻ (mM)
        self.h_concentration = cp.full((num_neurons,), 7.2)  # 내부 pH (H⁺ 농도)

    def update_potential(self, time_step=0.1):
        """막전위 업데이트"""
        self.potentials[self.na_channel_open] += 10.0 * time_step
        self.potentials[self.k_channel_open] -= 10.0 * time_step
        self.potentials[self.ca_channel_open] += 2.0 * time_step
        self.potentials[self.cl_channel_open] -= 5.0 * time_step
        self.potentials[self.h_channel_open] += 0.5 * time_step  # H⁺ 통로 개방 시 막전위 상승

        # 막전위 제한 (-80mV ~ 50mV)
        self.potentials = cp.clip(self.potentials, -80.0, 50.0)

    def update_ion_pumps(self):
        """🔹 이온 펌프 (Na⁺/K⁺, Ca²⁺, H⁺, 독성 제거)"""
        p_class_activity = 0.1  # P-class 펌프 활동 수준

        # 1. Na⁺/K⁺ 펌프 (나트륨-칼륨 펌프) - ATP 소모
        na_flux = p_class_activity * (self.na_out - self.na_in)
        k_flux = p_class_activity * (self.k_in - self.k_out)
        self.na_in += na_flux * 0.1
        self.na_out -= na_flux * 0.1
        self.k_in -= k_flux * 0.1
        self.k_out += k_flux * 0.1

        # 2. Ca²⁺ 펌프 (칼슘 이온 농도 조절)
        ca_pump_activity = 0.05
        ca_transport = ca_pump_activity * self.ca_in
        self.ca_in -= ca_transport
        self.ca_out += ca_transport

        # 3. H⁺ 펌프 (pH 조절)
        h_pump_activity = 0.02
        h_adjustment = h_pump_activity * (self.h_concentration - 7.2)
        self.h_concentration -= h_adjustment

        # 4. 독성 물질 제거 (ATP 사용)
        detox_activity = 0.01
        toxic_removal = detox_activity * self.potentials
        self.potentials -= toxic_removal * 0.1

    def update_channels(self):
        """🔹 이온 채널 활성화"""
        vm = self.potentials

        # Na⁺ 채널 업데이트
        na_prob = 1 / (1 + cp.exp(-(vm + 50.0) / 5.0))
        self.na_channel_open = cp.random.rand(self.num_neurons) < na_prob

        # K⁺ 채널 업데이트
        k_prob = 1 / (1 + cp.exp(-(vm) / 10.0))
        self.k_channel_open = cp.random.rand(self.num_neurons) < k_prob

        # Ca²⁺ 채널 업데이트
        ca_prob = 1 / (1 + cp.exp(-(vm + 40.0) / 5.0))
        self.ca_channel_open = cp.random.rand(self.num_neurons) < ca_prob

        # Cl⁻ 채널 업데이트
        cl_prob = 1 / (1 + cp.exp((vm + 60.0) / 5.0))
        self.cl_channel_open = cp.random.rand(self.num_neurons) < cl_prob

        # H⁺ 채널 업데이트 (pH 조절)
        h_prob = 1 / (1 + cp.exp(-(vm + 30.0) / 5.0))
        self.h_channel_open = cp.random.rand(self.num_neurons) < h_prob

    def update_refractory_period(self, time_step=0.1):
        """불응기 업데이트"""
        self.absolute_refractory_period = cp.clip(self.absolute_refractory_period - time_step, 0.0, None)

    def fire(self):
        """🔹 뉴런 발화 여부"""
        fired = self.potentials >= self.thresholds
        self.potentials[fired] = -70.0  # 발화 후 초기화
        self.absolute_refractory_period[fired] = 2.0  # 불응기 설정
        return fired

    def update(self, time_step=0.1):
        """🔹 전체 뉴런 상태 업데이트"""
        self.update_channels()
        self.update_potential(time_step)
        self.update_ion_pumps()
        self.update_refractory_period(time_step)

class Synapse:
    """뉴런 간 신호 전달을 위한 시냅스 클래스"""

    def __init__(self, pre_neuron, post_neuron, num_synapses):
        self.pre_neuron = pre_neuron  # 신호를 보내는 뉴런
        self.post_neuron = post_neuron  # 신호를 받는 뉴런
        self.num_synapses = num_synapses

        # 시냅스 가중치 초기화
        self.weights = cp.random.uniform(0.1, 1.0, size=(num_synapses,))

        # 신경전달물질 관련 변수
        self.neurotransmitter_concentration = cp.zeros(num_synapses)
        self.release_probability = cp.full((num_synapses,), 0.8)

    def release_neurotransmitters(self):
        """신경전달물질 방출"""
        release_condition = cp.random.uniform(0.0, 1.0, size=(self.num_synapses,)) < self.release_probability
        self.neurotransmitter_concentration[release_condition] += 1.0

    def activate_receptors(self):
        """수용체 활성화 -> 후 뉴런 전위 변화"""
        receptor_activation_condition = self.neurotransmitter_concentration > 0.5
        activated_synapses = cp.where(receptor_activation_condition)[0]

        for synapse in activated_synapses:
            delta_potential = self.weights[synapse] * 5.0
            self.post_neuron.potentials[synapse] += delta_potential

        # 신경전달물질 감소 (재흡수)
        self.neurotransmitter_concentration *= 0.5

    def update_weights(self, learning_rate=0.01):
        """시냅스 가중치 학습"""
        pre_fired = self.pre_neuron.fire()
        post_fired = self.post_neuron.fire()

        coincidence = cp.logical_and(pre_fired, post_fired)
        self.weights[coincidence] += learning_rate
        self.weights = cp.clip(self.weights, 0.0, 10.0)


class NLPModel:
    """자연어 처리 모델 (뉴런 & 시냅스 기반)"""
    def __init__(self):
        # 문자 집합 (알파벳 + 숫자 + 특수 문자)
        self.alphabet = list(string.ascii_uppercase) + list(string.digits) + ['_', '.', ',', '!', '?']
        self.num_tokens = len(self.alphabet)

        # 입력 뉴런 & 출력 뉴런 생성
        self.input_neurons = Neuron(self.num_tokens)
        self.output_neurons = Neuron(self.num_tokens)

        # 시냅스 연결
        self.synapse = Synapse(self.input_neurons, self.output_neurons, self.num_tokens)

    def tokenize(self, text):
        """텍스트를 문자 토큰(뉴런)으로 변환"""
        tokens = [char.upper() if char.upper() in self.alphabet else '_' for char in text]
        return tokens

    def encode(self, tokens):
        """토큰을 뉴런 인덱스로 변환"""
        return [self.alphabet.index(token) for token in tokens]

    def decode(self, indices):
        """뉴런 인덱스를 토큰(문자)으로 변환"""
        return ''.join([self.alphabet[i] for i in indices])

    def train(self, text, epochs=1000):
        """학습 (뉴런 및 시냅스 업데이트)"""
        tokens = self.tokenize(text)
        indices = self.encode(tokens)

        for epoch in range(epochs):
            for i in range(len(indices) - 1):
                current_idx = indices[i]
                next_idx = indices[i + 1]

                # 현재 문자 뉴런 활성화
                self.input_neurons.potentials[current_idx] = 50.0

                # 신호 전달
                self.synapse.release_neurotransmitters()
                self.synapse.activate_receptors()

                # 가중치 업데이트
                self.synapse.update_weights()

                # 뉴런 초기화
                self.input_neurons.potentials[:] = -70.0
                self.output_neurons.potentials[:] = -70.0

    def predict(self, text, length=10):
        """다음 문자 예측 (학습된 시냅스 가중치 기반)"""
        tokens = self.tokenize(text)
        indices = self.encode(tokens)

        output_text = tokens

        for _ in range(length):
            current_idx = indices[-1]

            # 현재 문자 뉴런 활성화
            self.input_neurons.potentials[current_idx] = 50.0

            # 신경전달물질 방출 및 뉴런 활성화
            self.synapse.release_neurotransmitters()
            self.synapse.activate_receptors()

            # 가장 강한 시냅스 가중치를 가진 뉴런 찾기
            weighted_potentials = self.output_neurons.potentials * self.synapse.weights
            next_idx = cp.argmax(weighted_potentials)

            output_text.append(self.alphabet[next_idx])
            indices.append(next_idx)

            # 뉴런 초기화
            self.input_neurons.potentials[:] = -70.0
            self.output_neurons.potentials[:] = -70.0

        return self.decode(indices)

    def self_supervised_learning(self, text, epochs=1000, mask_ratio=0.3):
        """자기 지도 학습"""
        tokens = self.tokenize(text)
        indices = self.encode(tokens)

        for epoch in range(epochs):
            masked_indices = indices.copy()
            mask_count = int(len(indices) * mask_ratio)
            mask_positions = cp.random.choice(len(indices), mask_count, replace=False)

            for pos in mask_positions:
                masked_indices[pos] = self.alphabet.index('_')

            for i in range(len(masked_indices) - 1):
                current_idx = masked_indices[i]
                next_idx = indices[i + 1]

                self.input_neurons.potentials[current_idx] = 50.0
                self.synapse.release_neurotransmitters()
                self.synapse.activate_receptors()

                predicted_idx = cp.argmax(self.output_neurons.potentials * self.synapse.weights)

                if predicted_idx == next_idx:
                    self.synapse.weights[current_idx] += 0.02
                else:
                    self.synapse.weights[current_idx] -= 0.01

        print("자기 지도 학습 완료!")

    def sequential_learning(self, text):
        """순차 학습"""
        self.train(text, epochs=1)
        print(f"입력 문장 '{text}' 학습 완료!")

    def batch_learning(self, texts, epochs=1000):
        """배치 학습"""
        for epoch in range(epochs):
            for text in texts:
                self.sequential_learning(text)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: 배치 학습 진행 중...")
        print("배치 학습 완료!")
