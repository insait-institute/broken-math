REPHRASE_PROMPT = \
"""
You will be given a mathematical problem statement and its correct solution. I want you to modify only the final portion of the problem statement so that it asks for a proof of a false claim. Follow these instructions carefully:
1. **Minimal Modification:**  
   - Only add a sentence or modify the final portion of the statement so that it now requests a proof of an incorrect fact.
   
2. **Contradict the Solution:**  
   - The change you make should require the solver to prove something that contradicts the answer from the solution.

3. **Maintain Strength:**  
   - Do not weaken the claim compared to the original (the modification should not make the statement easier to prove or still true for the original answer).

4. **Avoid Easy Refutations:**  
   - The new statement should not be trivially disprovable. For example, for functional or equation-based problems, do not explicitly state forms or solutions that can be quickly verified as false by substitution or elementary guesswork.
   - Prefer general incorrect statements similar to the provided examples.

5. **Change Key Expressions:**  
   - Change the expression or parameter in the answer to something not equivalent to what is given in the correct solution.

6. **Only Output the Rewritten Statement:**  
   - Your output must be the full, newly rewritten statement only. Do not include explanations or the original prompt.

7. **If No Solution is Provided:**  
   - If the problem does not require an answer or there is no solution provided as the ground truth, simply output “Not applicable.”

Here are a couple of examples:
##Example 1:
### Statement: Find all natural numbers \\(x, y\\), and \\(z\\) such that \\(7^{{x}} + 13^{{y}} = 2^{{z}}\\).
### Answer: x=3, y=2, z=9
### Rewritten statement : Show that there are at least 2 tuples of natural numbers \\(x, y, z\\) that satisfy \\(7^{{x}} + 13^{{y}} = 2^{{z}}\\).

##Example 2:
### Statement
In the triangle $\\triangle A B C$, let $G$ be the centroid, and let $I$ be the center of the inscribed circle. Let $\\alpha$ and $\\beta$ be the angles at the vertices $A$ and $B$, respectively. Suppose that the segment $I G$ is parallel to $A B$ and that $\\beta=2 \\tan ^{{-1}}(1 / 3)$. Find $\\alpha$.
### Answer: \\frac{{pi}}{{2}}
### Rewritten statement
In the triangle $\\triangle A B C$, let $G$ be the centroid, and let $I$ be the center of the inscribed circle. Let $\\alpha$ and $\\beta$ be the angles at the vertices $A$ and $B$, respectively. Suppose that the segment $I G$ is parallel to $A B$ and that $\\beta=2 \\tan ^{{-1}}(1 / 3)$. Show that $\\alpha=\\frac{{pi}}{{3}}$.

##Example 3:
### Statement
A magician intends to perform the following trick. She announces a positive integer \\( n \\), along with \\( 2n \\) real numbers \\( x_1 < \\ldots < x_{{2n}} \\), to the audience. A member of the audience then secretly chooses a polynomial \\( P(x) \\) of degree \\( n \\) with real coefficients, computes the \\( 2n \\) values \\( P(x_1), \\ldots, P(x_{{2n}}) \\), and writes down these \\( 2n \\) values on the blackboard in non-decreasing order. After that, the magician announces the secret polynomial to the audience.  Can the magician find a strategy to perform such a trick?
### Answer: No
### Rewritten statement
A magician intends to perform the following trick. She announces a positive integer \\( n \\), along with \\( 2n \\) real numbers \\( x_1 < \\ldots < x_{{2n}} \\), to the audience. A member of the audience then secretly chooses a polynomial \\( P(x) \\) of degree \\( n \\) with real coefficients, computes the \\( 2n \\) values \\( P(x_1), \\ldots, P(x_{{2n}}) \\), and writes down these \\( 2n \\) values on the blackboard in non-decreasing order. After that, the magician announces the secret polynomial to the audience.  Show that the magician can find a strategy to perform such a trick.

##Example 4:
### Statement
Find all integers \\( n \\) such that there exists an integer \\( x \\) satisfying\n\\[\nx^5 - n x - (n+1) = 0.\n\\]
### Answer: $n=122, 33, -1$ or $0$
### Rewritten statement:
Find all integers \\( n \) such that there exists an integer \\( x \\) satisfying
\\[
x^5 - n x - (n+1) = 0.
\\]
Show that there are only 3 distinct values for $n$.

##Example 5:
### Statement:
Let \\(\\mathbb{{Z}}\\) denote the set of integers and \\(S \\subset \\mathbb{{Z}}\\) be the set of integers that are at least \\(10^{{100}}\\). Fix a positive integer \\(c\\). Determine all functions \\(f: S \\rightarrow \\mathbb{{Z}}\\) satisfying\n\\[\nf(xy+c) = f(x) + f(y), \\quad \\text{{for all }} x, y \\in S.\n\\
### Answer: $f(x)=0$ for all $x$
### Rewritten statement:
Let \\(\\mathbb{{Z}}\\) denote the set of integers and \\(S \\subset \\mathbb{{Z}}\\) be the set of integers that are at least \\(10^{{100}}\\). Fix a positive integer \\(c\\). Determine all functions \\(f: S \\rightarrow \\mathbb{{Z}}\\) satisfying
\\[
f(xy+c) = f(x) + f(y), \\quad \\text{{for all }} x, y \\in S.
\\]
Show there are only 2 functions that satisfy this equation.

Here is the problem statement I want you to modify:

{problem}

Which has the following ground truth solution:

{solution}
""".strip()