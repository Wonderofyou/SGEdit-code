

box_proposal_instruction = '''# Your Role: Expert Bounding Box Adjustor

## Objective: Your task is to infer the new positions of the objects following the user instruction. 

## Scene Graph and Bounding Box Specifications
You will be provided with a fully decomposed scene graph, containing multiple (entity1, relation, entity2) pairs. The position of the entities in the scene is also know and given by the bounding boxes. Each bounding box should be represented as [x, y, w, h], where (x, y) are the coordinates of the top-left corner, and (w, h) are the with and height, respectively. All values should he between 0 and 1, indicating their relative position and size in the scene. 

## Example'''


add_examples = '''
- Example 1
Position Information: 
'knife': [0.10, 0.09, 0.22, 0.55] 
'plate': [0.20, 0.14, 0.60, 0.71] 
'drink': [0.84, 0.00, 0.16, 0.38]
Relation Information:
1. 'knife' -> on the left of -> 'plate'
2. 'drink' -> is adjacent to -> 'plate'
User Modification: 
Add an object: 'apple'. Add tuple: ['apple' -> on -> 'plate'] 
Reasoning: 
Since the apple is 'on' the plate, there should be overlapping between apple and plate.
Update Boxes: 
'apple': [0.30, 0.20, 0.3, 0.3]

- Example 2
Position Information:
'child': [0.38, 0.43, 0.27, 0.45]
'sofa': [0.17, 0.60, 0.64, 0.29]
'field': [0.00, 0.88, 1.00, 0.12]
Relation Information:
1. 'child' -> sitting on -> 'sofa'
2. 'sofa' -> on -> 'field'
User Modification: 
Add an object: 'vase'. Add tuple: ['vase' -> on -> 'field'] 
Reasoning:
Positioning the vase on the field, involves placing it on the field and reducing overlapping with existing objects like the child and the sofa. 
Update Boxes: 
'vase': [0.74, 0.60, 0.20, 0.30]

- Example 3
Position Information:
'kid': [0.06, 0.21, 0.27, 0.70]
'box': [0.37, 0.48, 0.63, 0.44]
Relation Information:
1. 'kid' -> is adjacent to -> 'box'
User Modification: 
Add an object: 'dog'. Add tuple: ['dog' -> beside -> 'kid'] 
Reasoning: 
Given the 'kid's bounding box starts at y = 0.21 and extends to 0.91, we infer the 'kid's feet to be around y = 0.91. For visual alignment, the 'dog's lower edge should also be approximately at y = 0.91 and it should be smaller than the 'kid'.
Update Boxes: 
'dog': [0.34, 0.65, 0.35, 0.26]

- Example 4
Position Information:
'person': [0.06, 0.21, 0.27, 0.70]
'bike': [0.37, 0.48, 0.63, 0.44]
'road': [0.01, 0.70, 0.99, 0.30]
Relation Information:
1. 'person' -> is adjacent to -> 'bike'
2. 'person' -> standing on -> 'road'
3. 'bike' -> on -> 'road'
User Modification: 
Add an object: 'helmet'. Add tuple: ['person' -> holding -> 'helmet'] 
Reasoning:
The helmet's placement needs to be close to the person's hands or at a spot visually suggesting that the person is holding it. Assuming that the hands are halfway up the body, approximately at y = 0.56, the helmet should be situated slightly above the hands.
Update Boxes: 
'helmet': [0.24, 0.50, 0.20, 0.20]

## Note:
1. Example 1 illustrates how to position the added object on the target location.
2. Example 2 illustrates placing the added object to reduce overlap with other objects.
2. Example 3 illustrates inferring the lower edge from the scene and the reasonable size of the object.
3. Examples 4 illustrate inferring the spatial position from their interaction.
4. It is preferable if the new added object have enough size, such as the height and width is larger than 0.2.

'''


replace_examples = '''
- Example 1
Position Information: 
'knife': [0.10, 0.09, 0.22, 0.55] 
'plate': [0.20, 0.14, 0.60, 0.71] 
'drink': [0.84, 0.00, 0.16, 0.38] 
Relation Information: 
1. 'knife' -> on the left of -> 'plate' 
2. 'drink' -> is adjacent to -> 'plate'
User Modification: 
Replace 'knife' by 'spoon' 
Reasoning: 
Replace the 'knife' with a 'spoon' in the scene graph by adjusting its position to match the knife's x-coordinate. Then, make the spoon shorter and wider to resemble its typical shape while keeping it to the left of the plate.
Update Boxes: 
'spoon': [0.10, 0.09, 0.25, 0.40]

- Example 2
Position Information:
'person': [0.06, 0.21, 0.27, 0.70]
'bike': [0.37, 0.48, 0.63, 0.44]
'road': [0.01, 0.70, 0.99, 0.30]
Relation Information:
1. 'person' -> is adjacent to -> 'bike'
2. 'person' -> standing on -> 'road'
3. 'bike' -> on -> 'road'
User Modification: 
Replace 'bike' by 'motorcycle' 
Reasoning: 
Since a motorcycle generally has a similar size as a bicycle, we can assuming that the editing will not change the bounding boxes.
Update Boxes: 
'motorcycle': [0.37, 0.48, 0.63, 0.44]

## Note:
1. Example 1 illustrates adjusting the object width and height, as the replacing object has a different size.
2. Example 2 illustrates keeping the object size the same, as the replacing object has a similar size.

'''


edge_examples = '''
- Example 1
Position Information:
'knife': [0.10, 0.09, 0.22, 0.55]
'plate': [0.20, 0.14, 0.60, 0.71]
'drink': [0.84, 0.00, 0.16, 0.38]
Relation Information:
1. 'knife' -> on the left of -> 'plate'
2. 'drink' -> is adjacent to -> 'plate'
User Modification: 
Modify a relationship: from ['knife' -> on the left of -> 'plate'] to ['knife' -> on the right of -> 'plate']
Reasoning: 
The plate's right edge is at x = 0.20 + 0.60 = 0.80. The knife's width is 0.22, and its height is 0.55. Place the knife's left edge at x = 0.78 to keep it visible and to the right of the plate. This might overlap with the drink, so move the knife down to y = 0.40 to resolve the overlap.
Update Boxes: 
'knife': [0.78, 0.40, 0.22, 0.55]

- Example 2
Position Information:
'person': [0.06, 0.21, 0.27, 0.70]
'bike': [0.37, 0.48, 0.63, 0.44]
'road': [0.01, 0.70, 0.99, 0.30]
Relation Information:
1. 'person' -> is adjacent to -> 'bike'
2. 'person' -> standing on -> 'road'
3. 'bike' -> on -> 'road'
User Modification: 
Modify a relationship: from ['person' -> is adjacent to -> 'bike'] to ['person' -> riding -> 'bike']
Reasoning: 
We can assume the person centered over the bike with a reduced height to indicate a seated, riding posture. This involves calculating a new top position for the person based on the bike's top position and possibly adjusting the left position to align the person more centrally over the bike.
Update Boxes: 
'person': [0.55, 0.43, 0.27, 0.37]

- Example 3
Position Information:
'child': [0.38, 0.43, 0.17, 0.35]
'sofa': [0.17, 0.60, 0.64, 0.29]
'field': [0.00, 0.88, 1.00, 0.12]
Relation Information:
1. 'child' -> sitting on -> 'sofa'
2. 'sofa' -> on -> 'field'
User Modification: 
Modify a relationship: from ['child' -> sitting on -> 'sofa'] to ['child' -> beside -> 'sofa']
Reasoning: 
Assuming 'beside' implies a side-by-side arrangement on the same horizontal plane, we'll position the child to the right of the sofa. The sofa's right edge is at x = 0.17 + 0.64 = 0.81. The child's width is 0.17, which falls within the boundary. Additionally, we can infer that the child changing from sitting to standing. Therefore, we'll move the child down towards the field and scale the height up to 0.40.
Update Boxes: 
'child': [0.81, 0.50, 0.17, 0.40]

- Example 4
Position Information:
'green car': [0.03, 0.23, 0.46, 0.67]
'blue truck': [0.54, 0.22, 0.43, 0.66]
Relation Information:
1. 'green car'-> is parked beside -> 'blue truck'
User Modification: 
Modify a relationship: from ['green car'-> is parked beside -> 'blue truck'] to ['green car'-> on the right of -> 'blue truck']
Reasoning: 
We need to move the 'green car' to the right of the 'blue truck'. The right edge of the blue truck is at x = 0.54+0.43 = 0.97, which indicate there is no enough space to position the green car to the right. The current location indicate that the 'green car' is on the left of the 'blue truck'. Therefore, we can try to swap the 'green car' and 'blue truck'.
Update Boxes: 
'green car': [0.44, 0.22, 0.35, 0.67], 'blue truck': [0.77, 0.23, 0.33, 0.66]


## Note
1. Example 1 illustrates keeping the object within the boundary, reducing overlap with other objects.
2. Example 2 illustrates repositioning the object to reflect a new relationship. Also, the size of the object is adjusted to reflect new interactions.
3. Example 3 illustrates repositioning the object to reflect a new relationship. Additional information from the scene graph is used to achieve a more realistic result.
4. Example 4 illustrates swapping the positions of objects when the spatial relationship between them is reversed. Otherwise, the object cannot fit within the boundary without significant overlap or drastic changes in size.

'''


single_instruction = '''Your task is to craft a simple text prompt based on provided context. This context outlines specific target objects and the scene around them. Below, you'll find examples.

## Examples
Example 1:
Target objects: "person" 
Scene information:
1. person -> seated on -> boat
2. boat -> on -> ocean
Expected prompt:
A photo of "person" is sitting on the boat.

Example 2:
Target objects: "apple" 
Scene information:
1. apple -> on the left of -> banana
Expected prompt:
A photo of "apple".

Example 3:
Target objects: "person" 
Scene information:
1. person -> beside -> bike
2. bike -> on -> street
Expected prompt:
A photo of "person" is standing on the street.

Example 4:
Target objects: "butterfly" 
Scene information:
1. butterfly -> above -> flower
Expected prompt:
A photo of "butterfly" flying.

## Note
1. Your prompt should follow the structure: "a photo of <target object> (action) (support surface)." Including the action and support surface is optional, depending on whether they are implied or explicitly mentioned in the scene details. Actions are applicable only to living entities like people and animals. 
2. You should focus on the given object. Exclude the irrelavant objects. A bad prompt like 'A photo of "apple" and "banana" on the table.

## Your task:
'''


multiple_instruction = '''Your task is to craft a simple text prompt based on provided context. This context outlines specific target objects and the scene around them. Below, you'll find examples to illustrate the expected format.

Example 1:
Target objects: "apple", "banana"
Scene information:
1. apple -> on the left of -> banana
Expected prompt:
A photo of "apple" and "banana".

Example 2:
Target objects: "person", "bike"
Scene information:
1. person -> on the left of -> bus
2. bike -> on -> street
3. person -> riding -> bike
Expected prompt:
A photo of "person" riding "bike".

Example 3:
Target objects: "person", "guitar"
Scene information:
1. person -> playing -> guitar
2. Person -> sitting on -> chair
Expected prompt:
A photo of "person" is playing "guitar".

Example 4:
Target objects: "person", "wallet "
Scene information:
Person -> holding -> wallet
Person -> standing on -> street
Expected prompt:
A photo of "person" is holding "wallet".

## Note
Make sure to include all specified target objects, drawing on the scene information to infer their relationships. 

## Your task:
'''

