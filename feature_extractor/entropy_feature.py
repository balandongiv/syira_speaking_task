
import numpy as np
def differential_entropy(channel_data):
    # print(f"Calculating differential entropy for channel data of shape {channel_data.shape}")
    variance = np.var(channel_data)
    if variance <= 0:
        return 0.0  # To handle zero variance to avoid log(0)
    return 0.5 * np.log(2 * np.pi * np.e * variance)

def approximate_entropy(channel_data, m=2, r=0.2):
    print(f"Calculating approximate entropy for channel data of shape {channel_data.shape}. This may take some time")
    def _count_patterns(data, m, r):
        n = len(data)
        count = 0
        for i in range(n - m + 1):
            pattern_i = data[i:i + m]
            for j in range(n - m + 1):
                if i != j:
                    pattern_j = data[j:j + m]
                    if np.max(np.abs(pattern_i - pattern_j)) <= r * np.std(data):
                        count += 1
        return count / (n - m + 1) / (n - m) if (n - m) > 0 else 0
    phi_m = _count_patterns(channel_data, m, r)
    phi_m_plus_1 = _count_patterns(channel_data, m + 1, r)
    return np.log(phi_m / phi_m_plus_1) if phi_m_plus_1 != 0 else np.nan