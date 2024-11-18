# Scene description
scene_desc_question = "What’s in this image? Describe the most distinctive feature for different people or objects in same category."

# Question to construct scene graphs
construct_sg_question='''Now you are given a task to build a scene graph from an input image. I will provide you some examples.

## Example
- Example 1:
Question1: What is in this image?
Answer1: Two men converse on a sidewalk, one in red with sunglasses facing the other in gray, both standing in the shade by a white building, with trees, a red car, and a bike rack nearby, in a sunny urban setting.
Question2: What are the main instances in this image?
Answer2: man in red shirt, man in gray jacket, sidewalk, trees, red car.
Question3: what is the relationship between the main instances? Given me the answer in scene graph format.
Answer3: 
    1. Man in red shirt -> standing on -> Sidewalk
    2. Man in gray jacket -> standing on -> Sidewalk
    3. Man in red shirt -> facing -> Man in gray jacket
    4. Trees -> growing from -> Sidewalk
    5. Red car -> parked on -> Sidewalk
    
- Example 2:
Question1: What is in this image?
Answer1: The image shows two umbrellas, one red and one black, inserted on a white sandy beach.
Question2: What are the main instances in this image?
Answer2: red umbrella, black umbrella, sandy beach.
Question3: What is the relationship between the main instances? Give me the answer in scene graph format.
Answer3: 
    1. Red umbrella -> placed on -> Sandy beach
    2. Black umbrella -> placed on -> Sandy beach
    3. Red umbrella -> adjacent to -> Black umbrella
    
- Example3:
Question1: What is in this image?
Answer1: The image shows an office setting. There is a person sitting on an office chair, facing away from the camera, working on a desktop computer with dual monitors. The desk is cluttered with various items including papers, a telephone, a bottle of water, a can, a coffee mug, and some office supplies. On the left, there's a bookshelf with many books, binders, and folders. 
Question2: What are the main instances in this image?
Answer2: person, chair, computer, desk.
Question3: what is the relationship between the main instances? Given me the answer in scene graph format.
Answer3: 
    1. Person -> sitting on -> Chair
    2. Chair -> in front of -> Desk
    3. Computer -> placed on -> Desk
    4. Person -> using -> Computer
    5. Bookshelf -> on the left of -> Desk

# Guidelines for answering the Question2.
1. Excluding accessories: Excluding accessories and detachable parts (such as saddles, bridles, ropes, helmets, hats, and T-shirts). For example, if there's a woman wearing gloves and a hat, only include 'woman'. Similarly, if there's a flower in a pot, simply refer to it as 'pot with flower'.
2. Specify Foreground Objects Individually: Describe objects in the foreground individually. Instead of "two umbrellas," specify "one red umbrella" and "one black umbrella."
3. Simplify the Background: In the background, only note the main elements. You don’t need every detail, just the essentials like "sidewalk, beach, wall," etc.

Now answer the Question2 and Question3 following on the guidelines'''


# Question to determined type of node
classify_template = '''Your task is to classify nouns into three categories: Animate object(0), Non-Animate object(1), Background object(2).
Animate objects encompass entities capable of movement, such as humans or animals. Toys designed with animate characteristics, like "plush ducks," "LEGO figures," and "figurines," fall into this category as well. Non-animate objects, such as "plants", "boats," "teapots," and "books," lack the capacity for action. Background objects, such as "sky," "glass," "wall," "beach," "building," and "floor," serve as elements within the backdrop.

## Example:
nouns: ["cat", "plant", "boat", "person", "sky", "pluff duck", "wall", "bench", "lego figure", "teapot"]
outputs: [0, 1, 1, 0, 2, 0, 2, 1, 0, 1]

## Note:
1. Animate object is classified into 0, Non-Animate object is classified into 1, background object is classified into 2.
2. Directly output the classification result.

## Your task:
nouns: '''


# Question to obtain descriptions
nonanimate_question='''
# Examples
ex1: A photo of a <asset0>. It is a vintage typewriter with black keys and a silver frame.
ex2: A photo of a <asset0>. It is a ceramic tea pot with a floral design.
ex3: A photo of a <asset0>. It is an electric guitar with a red and white body.
ex4: A photo of a <asset0>. It is a plush teddy bear with a red bow tie.

# Guideline:
1. Description Format: Please craft your descriptions with the following format: "A photo of a <asset0>. {Include details about the subject's distinctive features}.". DO NOT CHANGE <asset0>.
2. Background Information: Omit any details pertaining to the background. Focus solely on the subject.
3. Brevity and Clarity: Aim for concise descriptions that highlight unique attributes. Limit your description to no more than two sentences. Additionally, ensure the language is simple and straightforward.

# Output'''


animate_question='''
# Examples
ex1: A photo of a <asset0> looking outside. It has a white coat and green eyes.
ex2: A photo of a <asset0> riding a bicycle. He is wearing a bright yellow helmet, a reflective vest, and casual clothing.
ex3: A photo of a <asset0> holding a book on his hands. He is wearing a glasses and a cozy sweater.
ex4: A photo of a <asset0> standing for night skiing. He is in a light grey ski jacket, black pants, and helmet.

# Guideline:
1. Description Format: Please craft your descriptions with the following structure: "A photo of a <asset0> {depicting the subject's action or their current pose, not features or accessories}. {Include details about the subject's distinctive features or accessories}.". DO NOT CHANGE <asset0>.
2. Background Information: Omit any details pertaining to the background. Focus solely on the subject.
3. Brevity and Clarity: Aim for concise descriptions that highlight unique attributes. Limit your description to no more than two sentences.Additionally, ensure the language is simple and straightforward.

# Output'''

