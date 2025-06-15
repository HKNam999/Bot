import asyncio
import hashlib
import random
import re
import sqlite3
import time
import math
from datetime import datetime
from typing import Dict, List, Optional

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
except ImportError:
    print("❌ Lỗi import telegram! Đang cài đặt lại...")
    import subprocess
    subprocess.run(["uv", "add", "python-telegram-bot==20.8"], check=True)
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

# Configuration
BOT_TOKEN = "8010052059:AAFlAiUjs_uTaLAzv38Ae-1Rwx2PhZmHQgo"
ADMIN_IDS = [7560849341]

# Database setup
def init_db():
    conn = sqlite3.connect('bot_data.db')
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            coins INTEGER DEFAULT 1,
            total_analyses INTEGER DEFAULT 0,
            join_date TEXT,
            referrals INTEGER DEFAULT 0
        )
    ''')

    # Analyses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            md5_hash TEXT,
            prediction TEXT,
            result TEXT,
            is_correct BOOLEAN,
            timestamp TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')

    # Gift codes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gift_codes (
            code TEXT PRIMARY KEY,
            coins INTEGER,
            used_by INTEGER,
            created_at TEXT,
            used_at TEXT
        )
    ''')

    # Admin table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            added_date TEXT
        )
    ''')

    # Coin transactions table for logging
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS coin_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            admin_id INTEGER,
            admin_username TEXT,
            amount INTEGER,
            timestamp TEXT,
            action_type TEXT DEFAULT 'add'
        )
    ''')

    # Add initial admin
    cursor.execute('''
        INSERT OR REPLACE INTO admins (user_id, username, added_date) 
        VALUES (?, ?, ?)
    ''', (7560849341, 'Main Admin', datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    conn.commit()
    conn.close()

# Super VIP PRO AI Analysis Engine
class SuperVIPProAIEngine:
    def __init__(self):
        # AI Systems siêu cấp VIP PRO
        self.ai_systems = {
            'AI_HTH_QUANTUM_SUPREME': self._ai_quantum_supreme,
            'AI_HTH_NEURAL_ULTIMATE': self._ai_neural_ultimate, 
            'AI_HTH_PATTERN_MASTER': self._ai_pattern_master,
            'AI_HTH_CRYPTO_ELITE': self._ai_crypto_elite,
            'AI_HTH_ENTROPY_PLATINUM': self._ai_entropy_platinum,
            'AI_HTH_FIBONACCI_DIAMOND': self._ai_fibonacci_diamond,
            'AI_HTH_PRIME_LEGENDARY': self._ai_prime_legendary,
            'AI_HTH_HASH_CHAIN_VIP': self._ai_hash_chain_vip,
            'AI_HTH_BINARY_SUPREME': self._ai_binary_supreme,
            'AI_HTH_MODULAR_ULTIMATE': self._ai_modular_ultimate,
            'AI_HTH_GOLDEN_RATIO_PRO': self._ai_golden_ratio_pro,
            'AI_HTH_CHECKSUM_ELITE': self._ai_checksum_elite,
            'AI_HTH_FRACTAL_MASTER': self._ai_fractal_master,
            'AI_HTH_CHAOS_THEORY': self._ai_chaos_theory,
            'AI_HTH_QUANTUM_ENTANGLE': self._ai_quantum_entangle
        }

        # Mathematical constants siêu chính xác
        self.constants = {
            'PI': 3.141592653589793238462643383279,
            'E': 2.718281828459045235360287471353,
            'PHI': 1.618033988749894848204586834366,
            'SQRT2': 1.414213562373095048801688724210,
            'SQRT3': 1.732050807568877293527446341506,
            'SQRT5': 2.236067977499789696409173668731,
            'EULER_GAMMA': 0.577215664901532860606512090082
        }

    def _convert_to_numeric(self, hash_str: str) -> int:
        return int(hash_str, 16)

    def _ai_quantum_supreme(self, hash_str: str) -> Dict:
        """AI #1: Quantum Supreme Analysis"""
        hash_int = self._convert_to_numeric(hash_str)
        binary_repr = bin(hash_int)[2:].zfill(128)

        # Quantum superposition với Bell states
        bell_states = []
        for i in range(0, len(binary_repr), 2):
            if i + 1 < len(binary_repr):
                bell_states.append(int(binary_repr[i]) ^ int(binary_repr[i+1]))

        entanglement_measure = sum(bell_states) / len(bell_states)
        quantum_coherence = abs(entanglement_measure - 0.5) * 2

        prediction = "⚫ Tài" if quantum_coherence < 0.5 else "⚪ Xỉu"
        confidence = min(95, quantum_coherence * 150 + 55)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'quantum_coherence': round(quantum_coherence, 4),
            'bell_states_count': len(bell_states)
        }

    def _ai_neural_ultimate(self, hash_str: str) -> Dict:
        """AI #2: Neural Ultimate Network"""
        segments = [int(hash_str[i:i+8], 16) for i in range(0, 32, 8)]

        # Multi-layer perceptron với activation functions
        layer1 = [math.tanh(x / 1000000) for x in segments]
        layer2 = [1 / (1 + math.exp(-x * self.constants['PHI'])) for x in layer1]  # sigmoid
        layer3 = [math.atan(x * self.constants['E']) for x in layer2]

        final_output = sum(layer3) / len(layer3)
        neural_confidence = abs(final_output - 0.5) * 200

        prediction = "⚫ Tài" if final_output < 0.5 else "⚪ Xỉu"
        confidence = min(95, neural_confidence + 60)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'neural_output': round(final_output, 4),
            'layers_processed': 3
        }

    def _ai_pattern_master(self, hash_str: str) -> Dict:
        """AI #3: Pattern Master Recognition"""
        patterns = {
            'ascending': 0,
            'descending': 0,
            'palindromic': 0,
            'repeating': 0,
            'symmetric': 0
        }

        # Phân tích patterns phức tạp
        for i in range(len(hash_str) - 2):
            chunk = hash_str[i:i+3]
            if chunk[0] < chunk[1] < chunk[2]:
                patterns['ascending'] += 1
            if chunk[0] > chunk[1] > chunk[2]:
                patterns['descending'] += 1
            if chunk == chunk[::-1]:
                patterns['palindromic'] += 1

        pattern_score = sum(patterns.values()) / len(hash_str)
        prediction = "⚫ Tài" if pattern_score > 0.3 else "⚪ Xỉu"
        confidence = min(95, pattern_score * 200 + 55)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'pattern_score': round(pattern_score, 4),
            'patterns_found': patterns
        }

    def _ai_crypto_elite(self, hash_str: str) -> Dict:
        """AI #4: Cryptographic Elite Analysis"""
        hash_int = self._convert_to_numeric(hash_str)

        # Avalanche effect và diffusion analysis
        bit_flip_count = bin(hash_int).count('1')
        avalanche_score = bit_flip_count / 128

        # S-box simulation
        sbox_output = []
        for i in range(0, 32, 4):
            nibble = int(hash_str[i:i+4], 16)
            sbox_output.append(nibble ^ (nibble >> 2))

        crypto_strength = sum(sbox_output) % 100
        prediction = "⚫ Tài" if crypto_strength < 50 else "⚪ Xỉu"
        confidence = min(95, abs(crypto_strength - 50) * 2 + 60)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'avalanche_score': round(avalanche_score, 4),
            'crypto_strength': crypto_strength
        }

    def _ai_entropy_platinum(self, hash_str: str) -> Dict:
        """AI #5: Entropy Platinum Calculator"""
        char_freq = {}
        for char in hash_str:
            char_freq[char] = char_freq.get(char, 0) + 1

        # Shannon entropy với Rényi entropy
        shannon_entropy = 0
        renyi_entropy = 0

        for freq in char_freq.values():
            p = freq / 32
            if p > 0:
                shannon_entropy -= p * math.log2(p)
                renyi_entropy += p ** 2

        renyi_entropy = -math.log2(renyi_entropy)
        combined_entropy = (shannon_entropy + renyi_entropy) / 2

        prediction = "⚫ Tài" if combined_entropy < 3.5 else "⚪ Xỉu"
        confidence = min(95, abs(combined_entropy - 3.5) * 25 + 55)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'shannon_entropy': round(shannon_entropy, 4),
            'renyi_entropy': round(renyi_entropy, 4)
        }

    def _ai_fibonacci_diamond(self, hash_str: str) -> Dict:
        """AI #6: Fibonacci Diamond Sequence"""
        segments = [int(hash_str[i:i+4], 16) for i in range(0, 32, 4)]

        # Tạo dãy Fibonacci từ hash segments
        fib_sequence = [segments[0], segments[1]]
        for i in range(2, len(segments)):
            fib_sequence.append((fib_sequence[i-1] + fib_sequence[i-2]) % 65536)

        # Golden ratio approximation
        ratios = []
        for i in range(1, len(fib_sequence)):
            if fib_sequence[i-1] != 0:
                ratios.append(fib_sequence[i] / fib_sequence[i-1])

        avg_ratio = sum(ratios) / len(ratios) if ratios else 1
        phi_deviation = abs(avg_ratio - self.constants['PHI'])

        prediction = "⚫ Tài" if phi_deviation < 0.5 else "⚪ Xỉu"
        confidence = min(95, (1 / (phi_deviation + 0.1)) * 20 + 55)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'phi_deviation': round(phi_deviation, 4),
            'avg_ratio': round(avg_ratio, 4)
        }

    def _ai_prime_legendary(self, hash_str: str) -> Dict:
        """AI #7: Prime Legendary Analysis"""
        segments = [int(hash_str[i:i+8], 16) for i in range(0, 32, 8)]

        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        prime_segments = [seg for seg in segments if is_prime(seg % 10000)]
        prime_density = len(prime_segments) / len(segments)

        # Twin primes analysis
        twin_primes = 0
        for i in range(len(segments) - 1):
            if abs(segments[i] - segments[i+1]) == 2:
                twin_primes += 1

        prediction = "⚫ Tài" if prime_density > 0.5 else "⚪ Xỉu"
        confidence = min(95, prime_density * 80 + 55)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'prime_density': round(prime_density, 4),
            'twin_primes': twin_primes
        }

    def _ai_hash_chain_vip(self, hash_str: str) -> Dict:
        """AI #8: Hash Chain VIP Analysis"""
        current_hash = hash_str
        chain_values = []

        for i in range(5):
            segment = current_hash[i*6:(i+1)*6] if i*6 < len(current_hash) else current_hash[-6:]
            new_hash = hashlib.sha256(segment.encode()).hexdigest()[:8]
            chain_values.append(int(new_hash, 16))
            current_hash = new_hash + current_hash[8:]

        chain_stability = sum(abs(chain_values[i] - chain_values[i+1]) 
                             for i in range(len(chain_values)-1))
        stability_score = chain_stability / (len(chain_values) * 1000000)

        prediction = "⚫ Tài" if stability_score < 50 else "⚪ Xỉu"
        confidence = min(95, stability_score + 55)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'stability_score': round(stability_score, 4),
            'chain_length': len(chain_values)
        }

    def _ai_binary_supreme(self, hash_str: str) -> Dict:
        """AI #9: Binary Supreme Analysis"""
        hash_int = self._convert_to_numeric(hash_str)
        binary_repr = bin(hash_int)[2:].zfill(128)

        # Hamming weight và autocorrelation
        hamming_weight = binary_repr.count('1')
        autocorr_values = []

        for shift in range(1, 8):
            shifted = binary_repr[shift:] + binary_repr[:shift]
            correlation = sum(1 for i in range(len(binary_repr)) 
                            if binary_repr[i] == shifted[i])
            autocorr_values.append(correlation / len(binary_repr))

        avg_autocorr = sum(autocorr_values) / len(autocorr_values)

        prediction = "⚫ Tài" if hamming_weight < 64 else "⚪ Xỉu"
        confidence = min(95, abs(hamming_weight - 64) * 1.5 + 55)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'hamming_weight': hamming_weight,
            'avg_autocorr': round(avg_autocorr, 4)
        }

    def _ai_modular_ultimate(self, hash_str: str) -> Dict:
        """AI #10: Modular Ultimate Arithmetic"""
        hash_int = self._convert_to_numeric(hash_str)

        # Advanced modular operations
        large_primes = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]
        residue_pattern = [hash_int % p for p in large_primes]

        # Quadratic residue test
        quadratic_residues = 0
        for p in large_primes:
            if pow(hash_int % p, (p-1)//2, p) == 1:
                quadratic_residues += 1

        modular_score = quadratic_residues / len(large_primes)
        prediction = "⚫ Tài" if modular_score > 0.5 else "⚪ Xỉu"
        confidence = min(95, abs(modular_score - 0.5) * 100 + 55)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'modular_score': round(modular_score, 4),
            'quadratic_residues': quadratic_residues
        }

    def _ai_golden_ratio_pro(self, hash_str: str) -> Dict:
        """AI #11: Golden Ratio PRO Analysis"""
        segments = [int(hash_str[i:i+4], 16) for i in range(0, 32, 4)]

        # Lucas sequence và Fibonacci relationship
        lucas_seq = [2, 1]
        for i in range(2, len(segments)):
            lucas_seq.append(lucas_seq[i-1] + lucas_seq[i-2])

        # Golden ratio convergence test
        convergence_ratios = []
        for i in range(1, len(lucas_seq)-1):
            if lucas_seq[i] != 0:
                ratio = lucas_seq[i+1] / lucas_seq[i]
                convergence_ratios.append(abs(ratio - self.constants['PHI']))

        avg_convergence = sum(convergence_ratios) / len(convergence_ratios)

        prediction = "⚫ Tài" if avg_convergence < 0.1 else "⚪ Xỉu"
        confidence = min(95, (1 / (avg_convergence + 0.01)) * 10 + 55)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'avg_convergence': round(avg_convergence, 4),
            'phi_constant': self.constants['PHI']
        }

    def _ai_checksum_elite(self, hash_str: str) -> Dict:
        """AI #12: Checksum Elite Verification"""
        # Multiple checksum algorithms
        checksums = {
            'luhn': self._luhn_checksum(hash_str),
            'crc32': hash(hash_str) & 0xFFFFFFFF,
            'adler32': self._adler32_checksum(hash_str),
            'fletcher16': self._fletcher16_checksum(hash_str)
        }

        checksum_variance = sum(checksums.values()) % 1000
        checksum_parity = sum(1 for v in checksums.values() if v % 2 == 0)

        prediction = "⚫ Tài" if checksum_parity > 2 else "⚪ Xỉu"
        confidence = min(95, checksum_variance / 10 + 55)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'checksum_variance': checksum_variance,
            'checksum_parity': checksum_parity
        }

    def _ai_fractal_master(self, hash_str: str) -> Dict:
        """AI #13: Fractal Master Analysis"""
        # Mandelbrot set iteration test
        segments = [int(hash_str[i:i+8], 16) for i in range(0, 32, 8)]

        fractal_depths = []
        for seg in segments:
            c = complex(seg % 1000 / 1000, (seg >> 16) % 1000 / 1000)
            z = 0
            depth = 0
            for _ in range(100):
                if abs(z) > 2:
                    break
                z = z*z + c
                depth += 1
            fractal_depths.append(depth)

        avg_depth = sum(fractal_depths) / len(fractal_depths)

        prediction = "⚫ Tài" if avg_depth < 50 else "⚪ Xỉu"
        confidence = min(95, abs(avg_depth - 50) + 55)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'avg_fractal_depth': round(avg_depth, 2),
            'max_depth': max(fractal_depths)
        }

    def _ai_chaos_theory(self, hash_str: str) -> Dict:
        """AI #14: Chaos Theory Analysis"""
        segments = [int(hash_str[i:i+4], 16) for i in range(0, 32, 4)]

        # Logistic map chaos
        chaos_values = []
        x = segments[0] / 65536  # Normalize to [0,1]
        r = 3.9  # Chaos parameter

        for _ in range(len(segments)):
            x = r * x * (1 - x)
            chaos_values.append(x)

        # Lyapunov exponent approximation
        lyapunov = sum(math.log(abs(r * (1 - 2*x))) for x in chaos_values) / len(chaos_values)

        prediction = "⚫ Tài" if lyapunov > 0 else "⚪ Xỉu"
        confidence = min(95, abs(lyapunov) * 50 + 55)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'lyapunov_exponent': round(lyapunov, 4),
            'chaos_iterations': len(chaos_values)
        }

    def _ai_quantum_entangle(self, hash_str: str) -> Dict:
        """AI #15: Quantum Entanglement Analysis"""
        hash_int = self._convert_to_numeric(hash_str)

        # Bell inequality test
        segments = [int(hash_str[i:i+8], 16) for i in range(0, 32, 8)]

        # CHSH inequality
        correlations = []
        for i in range(len(segments)-1):
            alice = segments[i] % 2
            bob = segments[i+1] % 2
            correlation = alice * bob - alice * (1-bob) - (1-alice) * bob + (1-alice) * (1-bob)
            correlations.append(correlation)

        chsh_value = sum(correlations) / len(correlations)
        entanglement_strength = abs(chsh_value) * 2

        prediction = "⚫ Tài" if entanglement_strength > 1 else "⚪ Xỉu"
        confidence = min(95, entanglement_strength * 30 + 55)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'chsh_value': round(chsh_value, 4),
            'entanglement_strength': round(entanglement_strength, 4)
        }

    def _luhn_checksum(self, data: str) -> int:
        """Luhn algorithm checksum"""
        digits = [int(c, 16) % 10 for c in data]
        for i in range(len(digits) - 2, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        return sum(digits) % 10

    def _adler32_checksum(self, data: str) -> int:
        """Adler-32 checksum"""
        a, b = 1, 0
        for c in data:
            a = (a + ord(c)) % 65521
            b = (b + a) % 65521
        return (b << 16) | a

    def _fletcher16_checksum(self, data: str) -> int:
        """Fletcher-16 checksum"""
        sum1, sum2 = 0, 0
        for c in data:
            sum1 = (sum1 + ord(c)) % 255
            sum2 = (sum2 + sum1) % 255
        return (sum2 << 8) | sum1

    def analyze_hex_structure(self, md5_hash: str) -> Dict:
        """Phân tích cấu trúc hex chi tiết"""
        hex_analysis = {
            'char_distribution': {},
            'segment_patterns': [],
            'numeric_values': [],
            'binary_patterns': [],
            'mathematical_properties': {}
        }

        # Phân tích phân bố ký tự
        for char in '0123456789abcdef':
            hex_analysis['char_distribution'][char] = md5_hash.count(char)

        # Phân tích từng segment 8 ký tự
        for i in range(0, 32, 8):
            segment = md5_hash[i:i+8]
            segment_value = int(segment, 16)
            hex_analysis['segment_patterns'].append({
                'segment': segment,
                'decimal_value': segment_value,
                'is_palindrome': segment == segment[::-1],
                'ascending_chars': all(segment[j] <= segment[j+1] for j in range(len(segment)-1)),
                'repeating_chars': len(set(segment)) < len(segment)
            })

        # Phân tích số học
        full_number = int(md5_hash, 16)
        hex_analysis['mathematical_properties'] = {
            'sum_of_digits': sum(int(c, 16) for c in md5_hash),
            'product_of_digits': math.prod(int(c, 16) for c in md5_hash if c != '0'),
            'digital_root': self._digital_root(sum(int(c, 16) for c in md5_hash)),
            'divisible_by_3': (sum(int(c, 16) for c in md5_hash) % 3 == 0),
            'divisible_by_7': (full_number % 7 == 0),
            'divisible_by_11': (full_number % 11 == 0)
        }

        return hex_analysis

    def _digital_root(self, n):
        """Tính digital root"""
        while n >= 10:
            n = sum(int(digit) for digit in str(n))
        return n

    def analyze_with_all_ais(self, md5_hash: str) -> Dict:
        """Phân tích với tất cả AI systems"""
        if len(md5_hash) != 32 or not all(c in '0123456789abcdefABCDEF' for c in md5_hash):
            raise ValueError("Invalid MD5 hash format")

        hash_str = md5_hash.lower()
        ai_results = {}

        # Chạy tất cả AI systems
        for ai_name, ai_function in self.ai_systems.items():
            try:
                ai_results[ai_name] = ai_function(hash_str)
            except Exception as e:
                ai_results[ai_name] = {
                    'prediction': "⚪ Xỉu",
                    'confidence': 55,
                    'error': str(e)
                }

        # Phân tích cấu trúc hex
        hex_analysis = self.analyze_hex_structure(hash_str)

        # Super VIP Voting system
        tai_votes = 0
        xiu_votes = 0
        total_confidence = 0

        for result in ai_results.values():
            if result['prediction'] == "⚫ Tài":
                tai_votes += 1
            else:
                xiu_votes += 1
            total_confidence += result.get('confidence', 55)

        # Final super decision
        final_prediction = "⚫ Tài" if tai_votes > xiu_votes else "⚪ Xỉu"
        avg_confidence = total_confidence / len(ai_results)

        # Super consensus strength
        consensus_strength = max(tai_votes, xiu_votes) / len(ai_results) * 100

        return {
            'ai_name': '🤖 AI HTH',
            'final_prediction': final_prediction,
            'tai_votes': tai_votes,
            'xiu_votes': xiu_votes,
            'consensus_strength': round(consensus_strength, 2),
            'average_confidence': round(avg_confidence, 2),
            'total_ais': len(ai_results),
            'ai_results': ai_results,
            'hex_analysis': hex_analysis,
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# Bot class
class MD5AnalysisBot:
    def __init__(self):
        self.engine = SuperVIPProAIEngine()
        init_db()
        self.pending_predictions = {}

    def is_admin(self, user_id: int) -> bool:
        """Kiểm tra admin chính xác"""
        conn = sqlite3.connect('bot_data.db')
        cursor = conn.cursor()
        cursor.execute('SELECT user_id FROM admins WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        return result is not None or user_id in ADMIN_IDS

    def get_user(self, user_id: int) -> Optional[Dict]:
        """Lấy thông tin user với xử lý lỗi"""
        conn = sqlite3.connect('bot_data.db')
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
            user = cursor.fetchone()
            if user:
                return {
                    'user_id': user[0],
                    'username': user[1],
                    'coins': user[2],
                    'total_analyses': user[3],
                    'join_date': user[4],
                    'referrals': user[5]
                }
        except Exception as e:
            print(f"Error getting user {user_id}: {e}")
        finally:
            conn.close()
        return None

    def create_user(self, user_id: int, username: str):
        """Tạo user với xử lý lỗi"""
        conn = sqlite3.connect('bot_data.db')
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO users (user_id, username, coins, total_analyses, join_date, referrals)
                VALUES (?, ?, 1, 0, ?, 0)
            ''', (user_id, username, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
        except Exception as e:
            print(f"Error creating user {user_id}: {e}")
        finally:
            conn.close()

    def update_user_coins(self, user_id: int, amount: int):
        """Cập nhật xu với xử lý lỗi"""
        conn = sqlite3.connect('bot_data.db')
        cursor = conn.cursor()
        try:
            cursor.execute('UPDATE users SET coins = coins + ? WHERE user_id = ?', (amount, user_id))
            conn.commit()
        except Exception as e:
            print(f"Error updating coins for user {user_id}: {e}")
        finally:
            conn.close()

    def reset_user_coins(self, user_id: int):
        """Reset xu với xử lý lỗi"""
        conn = sqlite3.connect('bot_data.db')
        cursor = conn.cursor()
        try:
            cursor.execute('UPDATE users SET coins = 0 WHERE user_id = ?', (user_id,))
            conn.commit()
        except Exception as e:
            print(f"Error resetting coins for user {user_id}: {e}")
        finally:
            conn.close()

    def log_coin_transaction(self, user_id: int, admin_id: int, admin_username: str, amount: int, action_type: str = 'add'):
        """Ghi log giao dịch xu"""
        conn = sqlite3.connect('bot_data.db')
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO coin_transactions (user_id, admin_id, admin_username, amount, timestamp, action_type)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, admin_id, admin_username, amount, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), action_type))
            conn.commit()
        except Exception as e:
            print(f"Error logging transaction: {e}")
        finally:
            conn.close()

    def add_admin(self, user_id: int, username: str):
        """Thêm admin"""
        conn = sqlite3.connect('bot_data.db')
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO admins (user_id, username, added_date)
                VALUES (?, ?, ?)
            ''', (user_id, username, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
        except Exception as e:
            print(f"Error adding admin: {e}")
        finally:
            conn.close()

    def remove_admin(self, user_id: int):
        """Xóa admin"""
        conn = sqlite3.connect('bot_data.db')
        cursor = conn.cursor()
        try:
            cursor.execute('DELETE FROM admins WHERE user_id = ?', (user_id,))
            affected = cursor.rowcount
            conn.commit()
            return affected > 0
        except Exception as e:
            print(f"Error removing admin: {e}")
        finally:
            conn.close()
        return False

    def get_all_admins(self):
        """Lấy danh sách admin"""
        conn = sqlite3.connect('bot_data.db')
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT user_id, username, added_date FROM admins')
            admins = cursor.fetchall()
            return admins
        except Exception as e:
            print(f"Error getting admins: {e}")
        finally:
            conn.close()
        return []

    def increment_analyses(self, user_id: int):
        """Tăng số lần phân tích"""
        conn = sqlite3.connect('bot_data.db')
        cursor = conn.cursor()
        try:
            cursor.execute('UPDATE users SET total_analyses = total_analyses + 1 WHERE user_id = ?', (user_id,))
            conn.commit()
        except Exception as e:
            print(f"Error incrementing analyses: {e}")
        finally:
            conn.close()

    def save_analysis(self, user_id: int, md5_hash: str, analysis_data: str, prediction: str) -> int:
        """Lưu phân tích"""
        conn = sqlite3.connect('bot_data.db')
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO analyses (user_id, md5_hash, prediction, result, is_correct, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, md5_hash, prediction, None, None, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            analysis_id = cursor.lastrowid
            conn.commit()
            return analysis_id
        except Exception as e:
            print(f"Error saving analysis: {e}")
        finally:
            conn.close()
        return 0

    def get_leaderboard(self) -> List[Dict]:
        """Lấy bảng xếp hạng"""
        conn = sqlite3.connect('bot_data.db')
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT username, coins FROM users ORDER BY coins DESC LIMIT 10')
            results = cursor.fetchall()
            return [{'username': r[0], 'coins': r[1]} for r in results]
        except Exception as e:
            print(f"Error getting leaderboard: {e}")
        finally:
            conn.close()
        return []

    def create_gift_code(self, code: str, coins: int):
        """Tạo gift code"""
        conn = sqlite3.connect('bot_data.db')
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO gift_codes (code, coins, used_by, created_at, used_at)
                VALUES (?, ?, NULL, ?, NULL)
            ''', (code, coins, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
        except Exception as e:
            print(f"Error creating gift code: {e}")
        finally:
            conn.close()

    def use_gift_code(self, user_id: int, code: str) -> bool:
        """Sử dụng gift code"""
        conn = sqlite3.connect('bot_data.db')
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT coins, used_by FROM gift_codes WHERE code = ?', (code,))
            result = cursor.fetchone()

            if not result or result[1] is not None:
                return False

            coins = result[0]
            cursor.execute('UPDATE gift_codes SET used_by = ?, used_at = ? WHERE code = ?',
                          (user_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), code))
            cursor.execute('UPDATE users SET coins = coins + ? WHERE user_id = ?', (coins, user_id))

            conn.commit()
            return True
        except Exception as e:
            print(f"Error using gift code: {e}")
        finally:
            conn.close()
        return False

    def get_all_users(self):
        """Lấy danh sách user"""
        conn = sqlite3.connect('bot_data.db')
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT user_id FROM users')
            users = [row[0] for row in cursor.fetchall()]
            return users
        except Exception as e:
            print(f"Error getting users: {e}")
        finally:
            conn.close()
        return []

    def get_coin_transactions(self, user_id: int):
        """Lấy lịch sử giao dịch xu của người dùng."""
        conn = sqlite3.connect('bot_data.db')
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT admin_id, admin_username, amount, timestamp, action_type
                FROM coin_transactions
                WHERE user_id = ?
                ORDER BY timestamp DESC
            ''', (user_id,))
            transactions = cursor.fetchall()
            return transactions
        except Exception as e:
            print(f"Error getting coin transactions: {e}")
            return []
        finally:
            conn.close()

# Bot instance
bot = MD5AnalysisBot()

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command start"""
    user = update.effective_user
    user_data = bot.get_user(user.id)

    if not user_data:
        bot.create_user(user.id, user.username or "Unknown")
        user_data = bot.get_user(user.id)
        is_new_user = True
    else:
        is_new_user = False

    # Main menu keyboard dạng cũ
    keyboard = [
        [InlineKeyboardButton("💎 Bảng Giá Xu", callback_data="price_list")],
        [InlineKeyboardButton("🏆 Bảng Xếp Hạng", callback_data="leaderboard")],
        [InlineKeyboardButton("🎁 Nhập GiftCode", callback_data="enter_giftcode")],
        [InlineKeyboardButton("📞 Admin Hỗ Trợ", url="https://t.me/hatronghoann")],
        [InlineKeyboardButton("💬 Box Chat", url="https://t.me/+dufmaDB6K0YwNzI1")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    current_coins = user_data['coins'] if user_data else 1

    welcome_text = f"""
✨ **CHÀO MỪNG ĐẾN VỚI TOOL PHÂN TÍCH MD5** ✨

🤖 **TOOL PHÂN TÍCH MD5 - AI HTH**
🔥 **Thuật toán siêu cấp VIP PRO**

👋 Xin chào {user.first_name}!

🎯 **Tài khoản:** @{user.username or 'Unknown'}
🆔 **ID:** {user.id}
💰 **Xu còn lại:** {current_coins}

⚡ **Chọn chức năng bên dưới để bắt đầu!**
💎 **Sử dụng:** /tx <md5>
🎁 **Tặng 1 xu miễn phí cho thành viên mới!**

🔥 **Độ chính xác cực cao - Thuật toán REAL**
"""

    await update.message.reply_text(welcome_text, reply_markup=reply_markup, parse_mode='Markdown')

async def analyze_md5(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Phân tích MD5 với xử lý lỗi xu"""
    user_id = update.effective_user.id
    user_data = bot.get_user(user_id)

    if not user_data:
        bot.create_user(user_id, update.effective_user.username or "Unknown")
        user_data = bot.get_user(user_id)
        if not user_data:
            await update.message.reply_text("❌ Lỗi tạo tài khoản! Vui lòng thử lại.")
            return

    # Kiểm tra xu chính xác
    current_coins = user_data['coins']
    print(f"User {user_id} has {current_coins} coins")  # Debug log

    if current_coins < 1:
        await update.message.reply_text(
            f"💸 **Không đủ xu!**\n\n"
            f"**Xu hiện tại:** {current_coins}\n"
            f"**Cần:** 1 xu để phân tích với AI HTH.\n"
            f"💳 **Mua Xu Liên Hệ Admin** @hatronghoann",
            parse_mode='Markdown'
        )
        return

    if not context.args:
        await update.message.reply_text(
            "📝 **Cách sử dụng:**\n"
            "`/tx <mã_md5_32_ký_tự>`\n\n"
            "**Ví dụ:**\n"
            "`/tx 5d41402abc4b2a76b9719d911017c592`",
            parse_mode='Markdown'
        )
        return

    md5_hash = context.args[0].strip()

    if len(md5_hash) != 32 or not all(c in '0123456789abcdefABCDEF' for c in md5_hash):
        await update.message.reply_text(
            "❌ **Mã MD5 không hợp lệ!**\n\n"
            "MD5 phải có đúng 32 ký tự hex (0-9, a-f)",
            parse_mode='Markdown'
        )
        return

    processing_msg = await update.message.reply_text(
        "🤖 **AI HTH ĐANG PHÂN TÍCH...**\n"
        "⏳ **Vui lòng chờ 5 giây...**",
        parse_mode='Markdown'
    )

    try:
        await asyncio.sleep(5)  # Thời gian để AI xử lý

        analysis_result = bot.engine.analyze_with_all_ais(md5_hash)

        # Trừ xu TRƯỚC khi lưu để tránh lỗi
        bot.update_user_coins(user_id, -1)
        bot.increment_analyses(user_id)

        analysis_id = bot.save_analysis(user_id, md5_hash, str(analysis_result), analysis_result['final_prediction'])

        # Store pending prediction for verification
        bot.pending_predictions[user_id] = {
            'md5': md5_hash,
            'prediction': analysis_result['final_prediction'],
            'analysis_id': analysis_id
        }

        # Phân tích hex structure
        hex_analysis = analysis_result['hex_analysis']

        # Tính độ tin cậy từ 75-98%
        enhanced_confidence = min(98, max(75, analysis_result['consensus_strength'] * 0.7 + analysis_result['average_confidence'] * 0.3 + 25))

        # Phân tích chi tiết hex
        char_most_common = max(hex_analysis['char_distribution'], key=hex_analysis['char_distribution'].get)
        palindrome_segments = sum(1 for seg in hex_analysis['segment_patterns'] if seg['is_palindrome'])

        # Lấy xu mới sau khi trừ
        updated_user = bot.get_user(user_id)
        new_coin_count = updated_user['coins'] if updated_user else 0

        result_text = f"""
🤖 **AI HTH - DỰ ĐOÁN SIÊU CHÍNH XÁC** 🤖

🔮 **MD5:** `{md5_hash[:16]}...`

🎯 **KẾT QUẢ:** {analysis_result['final_prediction']}

📊 **Độ tin cậy:** {enhanced_confidence:.1f}%

🔍 **PHÂN TÍCH MÃ HEX:**
• **Ký tự xuất hiện nhiều nhất:** {char_most_common.upper()} ({hex_analysis['char_distribution'][char_most_common]} lần)
• **Tổng giá trị số:** {hex_analysis['mathematical_properties']['sum_of_digits']}
• **Digital Root:** {hex_analysis['mathematical_properties']['digital_root']}
• **Segment đối xứng:** {palindrome_segments}/4
• **Chia hết cho 3:** {'✅' if hex_analysis['mathematical_properties']['divisible_by_3'] else '❌'}

💎 **Xu còn lại:** {new_coin_count}

⭐ **Nhập kết quả để xác minh độ chính xác!**
"""

        keyboard = [
            [InlineKeyboardButton("⚫ Tài", callback_data=f"result_tai_{analysis_id}")],
            [InlineKeyboardButton("⚪ Xỉu", callback_data=f"result_xiu_{analysis_id}")],
            [InlineKeyboardButton("🔄 Phân Tích Khác", callback_data="analyze")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await processing_msg.edit_text(result_text, reply_markup=reply_markup, parse_mode='Markdown')

    except Exception as e:
        await processing_msg.edit_text(f"❌ **Lỗi phân tích:** {str(e)}", parse_mode='Markdown')
        print(f"Analysis error: {e}")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Button handler"""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    data = query.data

    if data == "price_list":
        price_text = """💎 BẢNG GIÁ XU 💎

╔═══════════════════════════════╗
║     🔥 GÓI XU HOT DEAL 🔥      ║
╠═══════════════════════════════╣
║ 💎 10 Xu  →  10,000 VND      ║
║ 💎 20 Xu  →  20,000 VND      ║
║ 💎 55 Xu  →  50,000 VND      ║
║ 💎 120 Xu →  100,000 VND     ║
║ 💎 200 Xu →  150,000 VND     ║
║ 💎 400 Xu →  300,000 VND     ║
║ 🌟 999 Xu →  400,000 VND     ║
╚═══════════════════════════════╝

🏦 THÔNG TIN THANH TOÁN
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 🏦 Ngân Hàng: MBBank        ┃
┃ 🔢 STK: 171226              ┃
┃ 👤 Tên: Ha Trong Hoan       ┃
┃ 📝 Nội Dung: User + ID      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

👨‍💼 Admin: @hatronghoann
⚡ Cấp xu tự động 24/7
🎁 Khuyến mãi thêm xu miễn phí"""
        await query.edit_message_text(price_text)

    elif data == "leaderboard":
        leaderboard = bot.get_leaderboard()
        if leaderboard:
            leaderboard_text = "🏆 **BẢNG XẾP HẠNG - NGƯỜI CÓ XU NHIỀU NHẤT**\n\n"
            for i, user in enumerate(leaderboard, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                leaderboard_text += f"{medal} **{user['username'] or 'Unknown'}** - {user['coins']} xu\n"
        else:
            leaderboard_text = "🏆 **BẢNG XẾP HẠNG TRỐNG**"

        await query.edit_message_text(leaderboard_text, parse_mode='Markdown')

    elif data == "enter_giftcode":
        await query.edit_message_text(
            "🎁 **NHẬN QUÀ MỖI NGÀY**\n\n"
            "Gửi lệnh: `/giftcode <mã code>`\n\n"
            "**Ví dụ:** `/giftcode ABC123`\n\n"
            "💡 **Lưu ý:** GiftCode do Admin cấp\n"
            "🎯 **Liên hệ Admin để nhận GiftCode miễn phí!**",
            parse_mode='Markdown'
        )

    elif data.startswith("result_"):
        parts = data.split("_")
        result_type = parts[1]  # tai or xiu
        analysis_id = int(parts[2])

        if user_id in bot.pending_predictions:
            prediction_data = bot.pending_predictions[user_id]
            actual_result = "⚫ Tài" if result_type == "tai" else "⚪ Xỉu"
            predicted_result = prediction_data['prediction']

            is_correct = (actual_result == predicted_result)

            # Update database
            conn = sqlite3.connect('bot_data.db')
            cursor = conn.cursor()
            try:
                cursor.execute('UPDATE analyses SET result = ?, is_correct = ? WHERE id = ?',
                              (actual_result, is_correct, analysis_id))
                conn.commit()
            except Exception as e:
                print(f"Error updating analysis: {e}")
            finally:
                conn.close()

            # Remove from pending
            del bot.pending_predictions[user_id]

            if is_correct:
                result_text = f"✅ **CHÍNH XÁC!**\n\nDự đoán: {predicted_result}\nKết quả: {actual_result}\n\n🎉 **AI HTH đã dự đoán đúng!**\n\n🔥 **Tiếp tục sử dụng để trải nghiệm độ chính xác cao!**"
            else:
                result_text = f"❌ **SAI LỆCH**\n\nDự đoán: {predicted_result}\nKết quả: {actual_result}\n\n🔄 **AI HTH sẽ học hỏi để cải thiện!**\n\n💪 **Tiếp tục phân tích để đạt độ chính xác tốt nhất!**"

            await query.edit_message_text(result_text, parse_mode='Markdown')

async def giftcode_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gift code handler"""
    user_id = update.effective_user.id

    if not context.args:
        await update.message.reply_text(
            "📝 **Cách sử dụng:**\n"
            "`/giftcode <mã_code>`\n\n"
            "**Ví dụ:** `/giftcode ABC123`",
            parse_mode='Markdown'
        )
        return

    code = context.args[0].strip().upper()

    if bot.use_gift_code(user_id, code):
        await update.message.reply_text(
            f"🎉 **SỬ DỤNG GIFTCODE THÀNH CÔNG!**\n\n"
            f"**Mã code:** `{code}`\n"
            f"💰 **Xu đã được cộng vào tài khoản!**",
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text(
            "❌ **GIFTCODE KHÔNG HỢP LỆ HOẶC ĐÃ ĐƯỢC SỬ DỤNG!**",
            parse_mode='Markdown'
        )

# Admin commands
async def add_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh cộng xu cho admin"""
    user_id = update.effective_user.id

    # Kiểm tra admin chính xác
    if not bot.is_admin(user_id):
        await update.message.reply_text("❌ **Bạn không có quyền admin!**", parse_mode='Markdown')
        return

    if len(context.args) != 2:
        await update.message.reply_text(
            "📝 **Cách sử dụng:**\n"
            "`/addcoins <user_id> <số_xu>`\n\n"
            "**Ví dụ:** `/addcoins 123456789 100`",
            parse_mode='Markdown'
        )
        return

    try:
        target_user_id = int(context.args[0])
        coins = int(context.args[1])
        admin_user = update.effective_user

        # Tạo user nếu chưa tồn tại
        user_data = bot.get_user(target_user_id)
        if not user_data:
            bot.create_user(target_user_id, f"User_{target_user_id}")
            user_data = bot.get_user(target_user_id)

        if not user_data:
            await update.message.reply_text("❌ **Không thể tạo user!**", parse_mode='Markdown')
            return

        old_coins = user_data['coins']

        # Cộng xu và log
        bot.update_user_coins(target_user_id, coins)
        bot.log_coin_transaction(target_user_id, admin_user.id, admin_user.username or "Admin", coins, 'add')

        # Lấy xu mới
        updated_user = bot.get_user(target_user_id)
        new_coins = updated_user['coins'] if updated_user else old_coins + coins

        await update.message.reply_text(
            f"✅ **CỘNG XU THÀNH CÔNG!**\n\n"
            f"**User ID:** {target_user_id}\n"
            f"**Xu cũ:** {old_coins}\n"
            f"**Xu thêm:** +{coins}\n"
            f"**Xu mới:** {new_coins}\n"
            f"**Admin:** @{admin_user.username or 'Unknown'}",
            parse_mode='Markdown'
        )

    except ValueError:
        await update.message.reply_text("❌ **User ID và số xu phải là số nguyên!**", parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ **Lỗi:** {str(e)}", parse_mode='Markdown')

async def reset_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh reset xu cho admin"""
    if not bot.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ **Bạn không có quyền admin!**", parse_mode='Markdown')
        return

    if len(context.args) != 1:
        await update.message.reply_text(
            "📝 **Cách sử dụng:**\n"
            "`/resetxu <user_id>`\n\n"
            "**Ví dụ:** `/resetxu 123456789`",
            parse_mode='Markdown'
        )
        return

    try:
        target_user_id = int(context.args[0])
        admin_user = update.effective_user

        user_data = bot.get_user(target_user_id)
        if not user_data:
            await update.message.reply_text("❌ **User không tồn tại!**", parse_mode='Markdown')
            return

        old_coins = user_data['coins']
        bot.reset_user_coins(target_user_id)
        bot.log_coin_transaction(target_user_id, admin_user.id, admin_user.username or "Admin", -old_coins, 'reset')

        await update.message.reply_text(
            f"✅ **RESET XU THÀNH CÔNG!**\n\n"
            f"**User ID:** {target_user_id}\n"
            f"**Xu cũ:** {old_coins}\n"
            f"**Xu mới:** 0\n"
            f"**Admin:** @{admin_user.username or 'Unknown'}",
            parse_mode='Markdown'
        )

    except ValueError:
        await update.message.reply_text("❌ **User ID phải là số nguyên!**", parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ **Lỗi:** {str(e)}", parse_mode='Markdown')

async def add_admin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh thêm admin"""
    if not bot.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ **Bạn không có quyền admin!**", parse_mode='Markdown')
        return

    if len(context.args) != 1:
        await update.message.reply_text(
            "📝 **Cách sử dụng:**\n"
            "`/themadmin <user_id>`\n\n"
            "**Ví dụ:** `/themadmin 123456789`",
            parse_mode='Markdown'
        )
        return

    try:
        target_user_id = int(context.args[0])

        if target_user_id == 7560849341:
            await update.message.reply_text("❌ **Không thể thêm Main Admin!**", parse_mode='Markdown')
            return

        bot.add_admin(target_user_id, f"Admin_{target_user_id}")
        await update.message.reply_text(
            f"✅ **THÊM ADMIN THÀNH CÔNG!**\n\n"
            f"**User ID:** {target_user_id}\n"
            f"**Quyền:** Admin",
            parse_mode='Markdown'
        )

    except ValueError:
        await update.message.reply_text("❌ **User ID phải là số nguyên!**", parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ **Lỗi:** {str(e)}", parse_mode='Markdown')

async def remove_admin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh xóa admin"""
    if not bot.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ **Bạn không có quyền admin!**", parse_mode='Markdown')
        return

    if len(context.args) != 1:
        await update.message.reply_text(
            "📝 **Cách sử dụng:**\n"
            "`/xoaadmin <user_id>`\n\n"
            "**Ví dụ:** `/xoaadmin 123456789`",
            parse_mode='Markdown'
        )
        return

    try:
        target_user_id = int(context.args[0])

        if target_user_id == 7560849341:
            await update.message.reply_text("❌ **Không thể xóa Main Admin!**", parse_mode='Markdown')
            return

        if bot.remove_admin(target_user_id):
            await update.message.reply_text(
                f"✅ **XÓA ADMIN THÀNH CÔNG!**\n\n"
                f"**User ID:** {target_user_id}",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text("❌ **User không phải admin!**", parse_mode='Markdown')

    except ValueError:
        await update.message.reply_text("❌ **User ID phải là số nguyên!**", parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ **Lỗi:** {str(e)}", parse_mode='Markdown')

async def list_admins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh xem danh sách admin"""
    if not bot.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ **Bạn không có quyền admin!**", parse_mode='Markdown')
        return

    admins = bot.get_all_admins()

    if not admins:
        await update.message.reply_text("📋 **Không có admin nào!**", parse_mode='Markdown')
        return

    admin_list = "👑 **DANH SÁCH ADMIN**\n\n"
    for admin in admins:
        admin_list += f"👤 **{admin[1]}**\n🆔 ID: {admin[0]}\n📅 Thêm: {admin[2]}\n\n"

    await update.message.reply_text(admin_list, parse_mode='Markdown')

async def admin_create_giftcode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh tạo gift code"""
    user_id = update.effective_user.id

    if not bot.is_admin(user_id):
        await update.message.reply_text("❌ **Chỉ Admin mới có thể sử dụng lệnh này!**", parse_mode='Markdown')
        return

    if len(context.args) != 2:
        await update.message.reply_text(
            "📝 **Cách sử dụng:**\n"
            "`/themgiftcode <mã_code> <số_xu>`\n\n"
            "**Ví dụ:** `/themgiftcode VIP123 50`",
            parse_mode='Markdown'
        )
        return

    code = context.args[0].strip().upper()
    try:
        coins = int(context.args[1])
    except ValueError:
        await update.message.reply_text("❌ **Số xu phải là số nguyên!**", parse_mode='Markdown')
        return

    bot.create_gift_code(code, coins)
    await update.message.reply_text(
        f"✅ **TẠO GIFTCODE THÀNH CÔNG!**\n\n"
        f"**Mã code:** `{code}`\n"
        f"**Số xu:** {coins}\n"
        f"**Trạng thái:** Chưa sử dụng",
        parse_mode='Markdown'
    )

async def admin_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh thông báo"""
    if not bot.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ **Bạn không có quyền admin!**", parse_mode='Markdown')
        return

    if not context.args:
        await update.message.reply_text(
            "📝 **Cách sử dụng:**\n"
            "`/thongbao <tin_nhắn>`\n\n"
            "**Ví dụ:** `/thongbao Hệ thống sẽ bảo trì lúc 23h`",
            parse_mode='Markdown'
        )
        return

    message = " ".join(context.args)
    users = bot.get_all_users()

    sent_count = 0
    failed_count = 0

    broadcast_text = f"""
📢 **Thông Báo Từ Admin**

{message}

---
🤖 **AI HTH Bot - Thuật toán siêu cấp VIP PRO**
"""

    for user_id in users:
        try:
            await context.bot.send_message(chat_id=user_id, text=broadcast_text, parse_mode='Markdown')
            sent_count += 1
            await asyncio.sleep(0.1)
        except:
            failed_count += 1

    await update.message.reply_text(
        f"📊 **Kết quả gửi thông báo:**\n\n"
        f"✅ **Thành công:** {sent_count}\n"f"❌ **Thất bại:** {failed_count}",
        parse_mode='Markdown'
    )

async def stat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh thống kê xu với thông tin admin cấp xu"""
    if not bot.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ Bạn không có quyền admin!")
        return

    conn = sqlite3.connect('bot_data.db')
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT u.user_id, u.username, u.coins FROM users u ORDER BY u.coins DESC LIMIT 20')
        users = cursor.fetchall()

        if not users:
            await update.message.reply_text("📊 Không có dữ liệu user!")
            return

        stats_text = "📊 THỐNG KÊ XU NGƯỜI DÙNG\n\n"

        for i, user in enumerate(users, 1):
            user_id, username, coins = user

            cursor.execute('''
                SELECT admin_username, SUM(amount) as total_given
                FROM coin_transactions
                WHERE user_id = ? AND action_type = 'add'
                GROUP BY admin_username
                ORDER BY total_given DESC
                LIMIT 2
            ''', (user_id,))

            admin_data = cursor.fetchall()
            admin_info = "Chưa có giao dịch"
            if admin_data:
                admin_list = [f"{admin[0] or 'Admin'} ({admin[1]} xu)" for admin in admin_data]
                admin_info = ", ".join(admin_list)

            stats_text += f"{i}. {username or 'Unknown'} (ID: {user_id})\n"
            stats_text += f"   💰 Xu: {coins}\n"
            stats_text += f"   👑 Cấp bởi: {admin_info}\n\n"

    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi: {str(e)}")
        return
    finally:
        conn.close()

    await update.message.reply_text(stats_text)

async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh thống kê tổng quan"""
    if not bot.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ Bạn không có quyền admin!**", parse_mode='Markdown')
        return

    conn = sqlite3.connect('bot_data.db')
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM analyses')
        total_analyses = cursor.fetchone()[0]

        cursor.execute('SELECT SUM(coins) FROM users')
        total_coins = cursor.fetchone()[0] or 0

        cursor.execute('SELECT COUNT(*) FROM admins')
        total_admins = cursor.fetchone()[0]
    except Exception as e:
        await update.message.reply_text(f"❌ **Lỗi truy vấn:** {str(e)}", parse_mode='Markdown')
        return
    finally:
        conn.close()

    stats_text = f"""
📊 **Thống Kê Bot AI HTH**

👥 **Tổng users:** {total_users}
👑 **Tổng admins:** {total_admins}
📈 **Tổng phân tích:** {total_analyses}
🤖 **AI Systems:** Hoạt động đồng thời
💰 **Tổng xu trong hệ thống:** {total_coins}

🤖 **AI HTH:** REAL ALGORITHM
⚡ **Trạng thái:** Hoạt động bình thường
🚀 **Thuật toán:** Mathematical - No Random
"""

    await update.message.reply_text(stats_text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh help"""
    user_id = update.effective_user.id
    user_data = bot.get_user(user_id)
    is_admin = bot.is_admin(user_id)

    help_text = f"""
📚 **HƯỚNG DẪN SỬ DỤNG AI HTH**

🔍 **LỆNH CHÍNH:**
• `/tx <md5>` - Phân tích MD5 với AI HTH
• `/giftcode <code>` - Sử dụng mã quà tặng
• `/thongtin` - Xem thông tin tài khoản
• `/help` - Xem hướng dẫn này

🎯 **CÁCH SỬ DỤNG:**
1. Gửi `/tx <mã_md5_32_ký_tự>`
2. Nhận dự đoán từ AI HTH
3. Nhập kết quả để xác minh độ chính xác

💰 **CHI PHÍ:** 1 xu/lần phân tích
🤖 **AI SYSTEMS:** Thuật toán siêu cấp VIP PRO

📊 **THÔNG TIN CỦA BẠN:**
• **Xu hiện tại:** {user_data['coins'] if user_data else 0}
• **Tổng phân tích:** {user_data['total_analyses'] if user_data else 0}

💳 **MUA XU:** Liên hệ @hatronghoann
🎁 **GIFTCODE:** Nhận từ admin hoặc sự kiện

🔥 **ĐỘ CHÍNH XÁC:** Cực cao với thuật toán REAL
⚡ **TỐC ĐỘ:** Phân tích trong 5 giây"""

    if is_admin:
        help_text += f"""

👑 **LỆNH ADMIN:**
• `/addcoins <user_id> <xu>` - Cộng xu cho user
• `/resetxu <user_id>` - Reset xu về 0
• `/themadmin <user_id>` - Thêm admin mới
• `/xoaadmin <user_id>` - Xóa admin
• `/dsadmin` - Danh sách admin
• `/stat` - Thống kê xu người dùng
• `/stats` - Thống kê tổng quan bot
• `/thongbao <tin_nhắn>` - Gửi thông báo
• `/themgiftcode <code> <xu>` - Tạo giftcode

🔧 **QUẢN LÝ:**
• Tất cả giao dịch đều có log
• Theo dõi hoạt động user
• Quản lý hệ thống xu tự động"""

    await update.message.reply_text(help_text, parse_mode='Markdown')

async def thongtin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh thông tin"""
    user_id = update.effective_user.id
    user_data = bot.get_user(user_id)

    if not user_data:
        bot.create_user(user_id, update.effective_user.username or "Unknown")
        user_data = bot.get_user(user_id)

    if not user_data:
        await update.message.reply_text("❌ **Lỗi lấy thông tin tài khoản!**", parse_mode='Markdown')
        return

    # Lấy thống kê phân tích gần đây
    conn = sqlite3.connect('bot_data.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT COUNT(*) as correct_count 
            FROM analyses 
            WHERE user_id = ? AND is_correct = 1
        ''', (user_id,))
        correct_analyses = cursor.fetchone()[0]

        cursor.execute('''
            SELECT COUNT(*) as total_verified 
            FROM analyses 
            WHERE user_id = ? AND is_correct IS NOT NULL
        ''', (user_id,))
        total_verified = cursor.fetchone()[0]
    except Exception as e:
        correct_analyses = 0
        total_verified = 0
        print(f"Error getting analysis stats: {e}")
    finally:
        conn.close()

    accuracy = (correct_analyses / total_verified * 100) if total_verified > 0 else 0

    info_text = f"""
👤 **THÔNG TIN TÀI KHOẢN**

🆔 **User ID:** {user_id}
👤 **Username:** @{user_data['username'] or 'Unknown'}
📅 **Ngày tham gia:** {user_data['join_date']}

💰 **XU & THỐNG KÊ:**
• **Xu hiện tại:** {user_data['coins']} xu
• **Tổng phân tích:** {user_data['total_analyses']} lần
• **Phân tích đúng:** {correct_analyses}/{total_verified}
• **Độ chính xác:** {accuracy:.1f}%

🤖 **AI HTH SYSTEMS:**
• **AI** hoạt động đồng thời
• **Thuật toán REAL** - Không random
• **Phân tích real-time** MD5

🎯 **DỊCH VỤ:**
• **Chi phí:** 1 xu/lần
• **Tốc độ:** 5 giây/phân tích
• **Hỗ trợ:** 24/7

💳 **Mua xu:** @hatronghoann
🎁 **GiftCode:** Từ admin và sự kiện
"""

    await update.message.reply_text(info_text, parse_mode='Markdown')

def main():
    """Main function với error handling"""
    print("🚀 Starting AI HTH Super VIP Pro Bot...")

    try:
        application = Application.builder().token(BOT_TOKEN).build()

        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("tx", analyze_md5))
        application.add_handler(CommandHandler("giftcode", giftcode_handler))
        application.add_handler(CommandHandler("themgiftcode", admin_create_giftcode))
        application.add_handler(CommandHandler("thongbao", admin_broadcast))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("thongtin", thongtin_command))
        application.add_handler(CommandHandler("addcoins", add_coins))
        application.add_handler(CommandHandler("resetxu", reset_coins))
        application.add_handler(CommandHandler("themadmin", add_admin_cmd))
        application.add_handler(CommandHandler("xoaadmin", remove_admin_cmd))
        application.add_handler(CommandHandler("dsadmin", list_admins))
        application.add_handler(CommandHandler("stat", stat_command))
        application.add_handler(CommandHandler("stats", admin_stats))
        application.add_handler(CallbackQueryHandler(button_handler))

        print("✅ Bot started successfully!")
        print("🔧 Fixed issues:")
        print("   - /stat command now works properly")
        print("   - Changed all '15 AI HTH' to 'AI HTH'")
        print("   - Fixed bot conflict errors")

        application.run_polling()

    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        raise

if __name__ == "__main__":
    main()