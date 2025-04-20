<h1 align="center">Fuzzy Logic Controller for Washing Machine</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.6+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen.svg" alt="Status">
</p>

<h2>📋 Overview</h2>

<p>
This project implements a Fuzzy Logic Controller (FLC) using Mamdani's approach to determine the optimal wash time for a domestic washing machine based on two input variables: dirt level and grease level.
</p>

<h2>🎯 Objective</h2>

<p>
Design and implement a fuzzy logic controller that determines appropriate wash times based on:
</p>
<ul>
  <li>Dirt level (0–100)</li>
  <li>Grease level (0–50)</li>
</ul>

<h2>🧠 Methodology</h2>

<h3>Fuzzification</h3>

<p><strong>Membership Functions:</strong> Triangular functions were used for fuzzification due to their simplicity and computational efficiency.</p>

<h4>Input Variables:</h4>

<table>
  <tr>
    <th colspan="2">Dirt (5 descriptors)</th>
  </tr>
  <tr>
    <td>Very Small Dirt (VSD)</td>
    <td>(0, 0, 25)</td>
  </tr>
  <tr>
    <td>Small Dirt (SD)</td>
    <td>(0, 25, 50)</td>
  </tr>
  <tr>
    <td>Medium Dirt (MD)</td>
    <td>(25, 50, 75)</td>
  </tr>
  <tr>
    <td>Heavy Dirt (HD)</td>
    <td>(50, 75, 100)</td>
  </tr>
  <tr>
    <td>Very Heavy Dirt (VHD)</td>
    <td>(75, 100, 100)</td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="2">Grease (3 descriptors)</th>
  </tr>
  <tr>
    <td>Small Grease (SG)</td>
    <td>(0, 0, 25)</td>
  </tr>
  <tr>
    <td>Medium Grease (MG)</td>
    <td>(0, 25, 50)</td>
  </tr>
  <tr>
    <td>Heavy Grease (HG)</td>
    <td>(25, 50, 50)</td>
  </tr>
</table>

<h4>Output Variable:</h4>

<table>
  <tr>
    <th colspan="2">Wash Time</th>
  </tr>
  <tr>
    <td>Very Short Time (VST)</td>
    <td>(0, 0, 15)</td>
  </tr>
  <tr>
    <td>Short Time (ST)</td>
    <td>(0, 15, 30)</td>
  </tr>
  <tr>
    <td>Medium Time (MT)</td>
    <td>(15, 30, 45)</td>
  </tr>
  <tr>
    <td>Long Time (HT)</td>
    <td>(30, 45, 60)</td>
  </tr>
  <tr>
    <td>Very Long Time (VHT)</td>
    <td>(45, 60, 60)</td>
  </tr>
</table>

<h3>Fuzzy Rule Base</h3>

<p>A rule table was implemented to map input combinations to output descriptors:</p>

<table>
  <tr>
    <th>Dirt\Grease</th>
    <th>SG</th>
    <th>MG</th>
    <th>HG</th>
  </tr>
  <tr>
    <td>VSD</td>
    <td>VST</td>
    <td>VST</td>
    <td>ST</td>
  </tr>
  <tr>
    <td>SD</td>
    <td>VST</td>
    <td>ST</td>
    <td>MT</td>
  </tr>
  <tr>
    <td>MD</td>
    <td>ST</td>
    <td>MT</td>
    <td>HT</td>
  </tr>
  <tr>
    <td>HD</td>
    <td>MT</td>
    <td>HT</td>
    <td>VHT</td>
  </tr>
  <tr>
    <td>VHD</td>
    <td>HT</td>
    <td>VHT</td>
    <td>VHT</td>
  </tr>
</table>

<h3>Fuzzy Inference</h3>

<p><strong>Max-Min Composition:</strong> For each input pair (dirt, grease), the activation strength of each rule was computed as the minimum of the corresponding membership values. The output membership function was clipped at this strength, and the results were aggregated using the maximum operator.</p>

<p>This approach allows the system to handle multiple rules firing simultaneously and combines their effects based on the strength of each rule's activation.</p>

<h3>Defuzzification</h3>

<p><strong>Center of Gravity (CoG):</strong> The crisp output (wash time) was calculated by integrating the aggregated output membership function over the range [0, 60] and dividing by the total area.</p>

<p>Formula:</p>

```
Wash Time = ∑(x · μaggregated(x)) / ∑(μaggregated(x))
```

<p>This was implemented numerically by discretizing the output range into 1000 points for accurate approximation of the continuous mathematical formula.</p>

<h2>💻 Implementation</h2>

<p>The fuzzy logic controller was implemented in Python. Key components include:</p>

```python
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
        'VHD': triangular(dirt, 75, 100, 100)
    }

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
```

<h3>How to Use</h3>

```python
python problem1_FLC.py
```

<p>Follow the prompts to enter dirt and grease levels to get the recommended wash time.</p>

<h2>📊 Results</h2>

<p>Example: For dirt = 40, grease = 30:</p>
<ul>
  <li>Output linguistic variable: Medium Time (MT)</li>
  <li>Crisp output value: 27.50 minutes</li>
</ul>

<p>The system successfully maps ranges of input values to appropriate wash times, dealing effectively with the uncertainty inherent in describing dirt and grease levels.</p>

<h2>⚙️ Requirements</h2>
<ul>
  <li>Python 3.6+</li>
  <li>NumPy</li>
</ul>

<hr>

