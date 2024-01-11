chatgpt_java = [
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
According to the method header, generate the method body in Java.
""",
        "0shot_user_prefix_1": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
According to the method header, generate the method body in Java.
""",
        "nshot_user_example": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
{body}
[END]
""",
        "nshot_user_prefix_1": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
""",
    },
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
According to the method header, generate the method body in Java.
""",
        "0shot_user_prefix_1": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
According to the method header, generate the method body in Java. 
First, I will give you some examples.
""",
        "nshot_user_example": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
{body}
[END]
""",
        "nshot_user_prefix_1": """
Then, try to generate the method body according to the method header and the above examples.
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
""",
    },
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
According to the method header, generate the method body in Java.
""",
        "0shot_user_prefix_1": """
[START]
{header}
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
According to the method header, generate the method body in Java.
""",
        "nshot_user_example": """
[START]
{header}
{body}
[END]
""",
        "nshot_user_prefix_1": """
[START]
{header}
""",
    },
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
According to the method header, generate the method body in Java.
""",
        "0shot_user_prefix_1": """
[START]
{header}
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
According to the method header, generate the method body in Java.
First, I will give you some examples.
""",
        "nshot_user_example": """
[START]
{header}
{body}
[END]
""",
        "nshot_user_prefix_1": """
Then, try to generate the method body according to the method header and the above examples.
[START]
{header}
""",
    },
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
Your task is to generate the method body according to the method header in Java. \
`[START]` and `[END]` represent the beginning and end of each method, respectively. \
You should only output the plain text of the method body and ends with `[END]`. \
Do not generate any comments or explanations.
""",
        "0shot_user_prefix_1": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
Your task is to generate the method body according to the method header in Java. \
`[START]` and `[END]` represent the beginning and end of each method, respectively. \
You should only output the plain text of the method body and ends with `[END]`. \
Do not generate any comments or explanations.
""",
        "nshot_user_example": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
{body}
[END]
""",
        "nshot_user_prefix_1": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
""",
    },
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
Your task is to generate the method body according to the method header in Java. \
`[START]` and `[END]` represent the beginning and end of each method, respectively. \
You should only output the plain text of the method body and ends with `[END]`. \
Do not generate any comments or explanations.
""",
        "0shot_user_prefix_1": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
Your task is to generate the method body according to the method header in Java. \
`[START]` and `[END]` represent the beginning and end of each method, respectively. \
You should only output the plain text of the method body and ends with `[END]`. \
Do not generate any comments or explanations. 
First, I will give you some examples.
""",
        "nshot_user_example": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
{body}
[END]
""",
        "nshot_user_prefix_1": """
Then, try to generate the method body according to the method header and the above examples.
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
""",
    },
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
Your task is to generate the method body according to the method header in Java. \
`[START]` and `[END]` represent the beginning and end of each method, respectively. \
You should only output the plain text of the method body and ends with `[END]`. \
Do not generate any comments or explanations.
""",
        "0shot_user_prefix_1": """
[START]
{header}
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
Your task is to generate the method body according to the method header in Java. \
`[START]` and `[END]` represent the beginning and end of each method, respectively. \
You should only output the plain text of the method body and ends with `[END]`. \
Do not generate any comments or explanations.
""",
        "nshot_user_example": """
[START]
{header}
{body}
[END]
""",
        "nshot_user_prefix_1": """
[START]
{header}
""",
    },
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
Your task is to generate the method body according to the method header in Java. \
`[START]` and `[END]` represent the beginning and end of each method, respectively. \
You should only output the plain text of the method body and ends with `[END]`. \
Do not generate any comments or explanations.
""",
        "0shot_user_prefix_1": """
[START]
{header}
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
Your task is to generate the method body according to the method header in Java. \
`[START]` and `[END]` represent the beginning and end of each method, respectively. \
You should only output the plain text of the method body and ends with `[END]`. \
Do not generate any comments or explanations.
First, I will give you some examples.
""",
        "nshot_user_example": """
[START]
{header}
{body}
[END]
""",
        "nshot_user_prefix_1": """
Then, try to generate the method body according to the method header and the above examples.
[START]
{header}
""",
    },
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
""",
        "0shot_user_prefix_1": """
[START]
{header}
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
""",
        "nshot_user_example": """
[START]
{header}
{body}
[END]
""",
        "nshot_user_prefix_1": """
[START]
{header}
""",
    },
]


chatgpt_py = [
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
According to the method header, generate the method body in Python.
""",
        "0shot_user_prefix_1": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
According to the method header, generate the method body in Python.
""",
        "nshot_user_example": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
{body}
[END]
""",
        "nshot_user_prefix_1": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
""",
    },
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
According to the method header, generate the method body in Python.
""",
        "0shot_user_prefix_1": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
According to the method header, generate the method body in Python. 
First, I will give you some examples.
""",
        "nshot_user_example": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
{body}
[END]
""",
        "nshot_user_prefix_1": """
Then, try to generate the method body according to the method header and the above examples.
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
""",
    },
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
According to the method header, generate the method body in Python.
""",
        "0shot_user_prefix_1": """
[START]
{header}
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
According to the method header, generate the method body in Python.
""",
        "nshot_user_example": """
[START]
{header}
{body}
[END]
""",
        "nshot_user_prefix_1": """
[START]
{header}
""",
    },
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
According to the method header, generate the method body in Python.
""",
        "0shot_user_prefix_1": """
[START]
{header}
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
According to the method header, generate the method body in Python.
First, I will give you some examples.
""",
        "nshot_user_example": """
[START]
{header}
{body}
[END]
""",
        "nshot_user_prefix_1": """
Then, try to generate the method body according to the method header and the above examples.
[START]
{header}
""",
    },
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
Your task is to generate the method body according to the method header in Python. \
`[START]` and `[END]` represent the beginning and end of each method, respectively. \
You should only output the plain text of the method body and ends with `[END]`. \
Do not generate any comments or explanations.
""",
        "0shot_user_prefix_1": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
Your task is to generate the method body according to the method header in Python. \
`[START]` and `[END]` represent the beginning and end of each method, respectively. \
You should only output the plain text of the method body and ends with `[END]`. \
Do not generate any comments or explanations.
""",
        "nshot_user_example": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
{body}
[END]
""",
        "nshot_user_prefix_1": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
""",
    },
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
Your task is to generate the method body according to the method header in Python. \
`[START]` and `[END]` represent the beginning and end of each method, respectively. \
You should only output the plain text of the method body and ends with `[END]`. \
Do not generate any comments or explanations.
""",
        "0shot_user_prefix_1": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
Your task is to generate the method body according to the method header in Python. \
`[START]` and `[END]` represent the beginning and end of each method, respectively. \
You should only output the plain text of the method body and ends with `[END]`. \
Do not generate any comments or explanations. 
First, I will give you some examples.
""",
        "nshot_user_example": """
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
{body}
[END]
""",
        "nshot_user_prefix_1": """
Then, try to generate the method body according to the method header and the above examples.
[START]
### METHOD_HEADER:
{header}
### METHOD_BODY:
""",
    },
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
Your task is to generate the method body according to the method header in Python. \
`[START]` and `[END]` represent the beginning and end of each method, respectively. \
You should only output the plain text of the method body and ends with `[END]`. \
Do not generate any comments or explanations.
""",
        "0shot_user_prefix_1": """
[START]
{header}
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
Your task is to generate the method body according to the method header in Python. \
`[START]` and `[END]` represent the beginning and end of each method, respectively. \
You should only output the plain text of the method body and ends with `[END]`. \
Do not generate any comments or explanations.
""",
        "nshot_user_example": """
[START]
{header}
{body}
[END]
""",
        "nshot_user_prefix_1": """
[START]
{header}
""",
    },
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
Your task is to generate the method body according to the method header in Python. \
`[START]` and `[END]` represent the beginning and end of each method, respectively. \
You should only output the plain text of the method body and ends with `[END]`. \
Do not generate any comments or explanations.
""",
        "0shot_user_prefix_1": """
[START]
{header}
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
Your task is to generate the method body according to the method header in Python. \
`[START]` and `[END]` represent the beginning and end of each method, respectively. \
You should only output the plain text of the method body and ends with `[END]`. \
Do not generate any comments or explanations.
First, I will give you some examples.
""",
        "nshot_user_example": """
[START]
{header}
{body}
[END]
""",
        "nshot_user_prefix_1": """
Then, try to generate the method body according to the method header and the above examples.
[START]
{header}
""",
    },
    {
        "0shot_system": """
""",
        "0shot_user_prefix_0": """
""",
        "0shot_user_prefix_1": """
[METHOD_HEADER]
{header}
[METHOD_BODY]
""",
        "nshot_system": """
""",
        "nshot_user_prefix_0": """
""",
        "nshot_user_example": """
[METHOD_HEADER]
{header}
[METHOD_BODY]
{body}
""",
        "nshot_user_prefix_1": """
[METHOD_BODY]
{header}
""",
    },
]
