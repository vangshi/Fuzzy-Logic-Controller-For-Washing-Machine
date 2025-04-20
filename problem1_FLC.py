import numpy as np

# Triangular membership function
def triangular(x, a, b, c):
    if a < x < b:
        return (x - a) / (b - a)
    elif b <= x < c:
        return (c - x) / (c - b)
    elif x == b:
        return 1.0
    else:
        return 0.0

# Membership functions for dirt (0–100)
def dirt_membership(dirt):
    return {
        'VSD': triangular(dirt, 0, 0, 25),
        'SD' : triangular(dirt, 0, 25, 50),
        'MD' : triangular(dirt, 25, 50, 75),
        'HD' : triangular(dirt, 50, 75, 100),
        'VHD': triangular(dirt, 751, 100, 100)
    }

# Membership functions for grease (0–50)
def grease_membership(grease):
    return {
        'SG': triangular(grease, 0, 0, 25),
        'MG': triangular(grease, 0, 25, 50),
        'HG': triangular(grease, 25, 50, 50)
    }

# Output membership functions for wash time (0–60)
output_membership_shapes = {
    'VST': (0, 0, 15),
    'ST' : (0, 15, 30),
    'MT' : (15, 30, 45),
    'HT' : (30, 45, 60),
    'VHT': (45, 60, 60)
}

# Rule base
rule_base = {
    ('VSD', 'SG'): 'VST', ('VSD', 'MG'): 'VST', ('VSD', 'HG'): 'ST',
    ('SD', 'SG') : 'VST', ('SD', 'MG') : 'ST' , ('SD', 'HG') : 'MT',
    ('MD', 'SG') : 'ST' , ('MD', 'MG') : 'MT' , ('MD', 'HG') : 'HT',
    ('HD', 'SG') : 'MT' , ('HD', 'MG') : 'HT' , ('HD', 'HG') : 'VHT',
    ('VHD', 'SG'): 'HT' , ('VHD', 'MG'): 'VHT', ('VHD', 'HG'): 'VHT'
}

# Aggregate the fuzzy output using max-min inference
def infer_output(dirt, grease):
    dirt_mf = dirt_membership(dirt)
    grease_mf = grease_membership(grease)

    output_activation = {
        'VST': 0.0,
        'ST' : 0.0,
        'MT' : 0.0,
        'HT' : 0.0,
        'VHT': 0.0
    }

    for dirt_label, dirt_value in dirt_mf.items():
        for grease_label, grease_value in grease_mf.items():
            rule = rule_base.get((dirt_label, grease_label))
            if rule:
                activation = min(dirt_value, grease_value)
                output_activation[rule] = max(output_activation[rule], activation)

    return output_activation

# Defuzzification using Center of Gravity (CoG)
def defuzzify(output_activation):
    x_vals = np.linspace(0, 60, 1000)
    numerator = 0.0
    denominator = 0.0

    for x in x_vals:
        max_membership = 0.0
        for label, (a, b, c) in output_membership_shapes.items():
            membership = triangular(x, a, b, c)
            clipped = min(membership, output_activation[label])
            max_membership = max(max_membership, clipped)

        numerator += x * max_membership
        denominator += max_membership

    return numerator / denominator if denominator != 0 else 0

# Complete function
def get_wash_time(dirt, grease):
    output_activation = infer_output(dirt, grease)
    wash_time = defuzzify(output_activation)
    return wash_time

if __name__ == "__main__":
    dirt_input = float(input("Enter dirt level (0–100): "))
    grease_input = float(input("Enter grease level (0–50): "))

    wash_time = get_wash_time(dirt_input, grease_input)
    print(f"\nWash Time: {wash_time:.2f} minutes")
