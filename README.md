# Adv_Applied_Machine_Learning
Data Science and Machine Learning work from DATA 410 data science capstone.

## We Love Sycamore (and Markdown)
<img src="https://user-images.githubusercontent.com/85187154/151677823-c1c41297-c130-49c8-96bc-7c2fe2a9719d.png" width=45% height=45% align='right'>

### This is Google's Sycamore quantum processor attached to its cryostat
It's very cool and uses cool math like this equation 2 + 2 = _The Schrodinger Equation_

### It also uses cool code like this
```python
def binary_half_adder(num_1, num_2):
    if len(num_1) != len(num_2):
        return 'mismatch lengths'
        
    else:
        reg_1 = qiskit.QuantumRegister(len(num_1), 'in_1')
        reg_2 = qiskit.QuantumRegister(len(num_2), 'in_2')
        reg_out = qiskit.QuantumRegister(len(num_2), 'out')
        reg_c = qiskit.ClassicalRegister(len(num_1), 'c')
        adder = qiskit.QuantumCircuit(reg_1,reg_2,reg_out,reg_c)
        
        for i in range(len(num_1)):
            if num_1[i] == '1':
                adder.x(reg_1[i])
            if num_2[i] == '1':
                adder.x(reg_2[i])
                
        #Make the 2 bit adders:
        adder.barrier()
        for q in range(len(reg_1)):
            adder.cx([reg_1[q],reg_2[q]],[reg_out[q],reg_out[q]])
            adder.ccx(reg_1[q], reg_2[q], reg_out[q])
            adder.barrier()
                    
        #Measurements
        for q in range(len(reg_out)):
            adder.measure(reg_out[q],reg_c[q])
        
        display(adder.draw(output='mpl'))
        #simulate and get output
        sim = qiskit.Aer.get_backend('aer_simulator')
        result = sim.run(adder).result() 
        counts = result.get_counts()
    return int(list(counts.keys())[0], 2)

binary_half_adder('10', '01')
```
<img src="https://user-images.githubusercontent.com/85187154/151679018-880d3534-72c8-498e-ade6-93bf810f51c3.png" width=45% height=45%>

This is a function I made in qiskit that makes a binary half adder circuit and outputs the result as a base 10 integer. Pretty cool! No superpositions in this one though so it's basically classical (boringg). This specific circuit is adding bx01 and bx10 to get bx11 or 3!
