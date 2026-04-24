from transformers import AutoTokenizer, AutoModelForCausalLM

import torch



model_id = "bigscience/bloomz-3b"

print(f"Loading {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")



# זה הפרומפט המדויק מהשגיאה שהעלת קודם (התשובה הנכונה לקרח פה היא 15)

prompt = "The towel is in box_29, the magnet is in box_04, the engine is in box_40, the drill is in box_51, the leaf is in box_34, the creature is in box_08, the chemical is in box_27, the tie is in box_72, the pot is in box_91, the seed is in box_97, the fork is in box_93, the dryer is in box_83, the magazine is in box_63, the rose is in box_50, the pipe is in box_82, the file is in box_58, the plate is in box_18, the scissors is in box_33, the mop is in box_17, the bill is in box_31, the clock is in box_71, the bomb is in box_68, the key is in box_85, the nail is in box_74, the television is in box_54, the bread is in box_76, the brush is in box_96, the shovel is in box_46, the document is in box_28, the vacuum is in box_81, the computer is in box_65, the basket is in box_87, the pan is in box_11, the egg is in box_06, the fig is in box_14, the ladder is in box_19, the ball is in box_20, the broom is in box_86, the brain is in box_43, the hammer is in box_89, the milk is in box_38, the kettle is in box_98, the screw is in box_24, the phone is in box_57, the shell is in box_59, the fish is in box_99, the string is in box_77, the plant is in box_16, the bell is in box_35, the tea is in box_00, the crown is in box_61, the shifter is in box_92, the sheet is in box_07, the watch is in box_49, the drink is in box_95, the gift is in box_70, the picture is in box_41, the card is in box_21, the medicine is in box_47, the boot is in box_88, the note is in box_60, the block is in box_10, the guitar is in box_75, the letter is in box_62, the bucket is in box_52, the dress is in box_32, the book is in box_67, the needle is in box_45, the soap is in box_66, the bus is in box_39, the coat is in box_78, the wire is in box_09, the drug is in box_26, the blender is in box_37, the stone is in box_73, the apple is in box_64, the game is in box_80, the pillow is in box_84, the machine is in box_53, the comb is in box_05, the wheel is in box_44, the cross is in box_25, the rock is in box_36, the flower is in box_55, the ice is in box_15, the branch is in box_56, the painting is in box_01, the camera is in box_30, the coffee is in box_22, the newspaper is in box_03, the blanket is in box_69, the thread is in box_90, the brick is in box_13, the hat is in box_79, the plane is in box_12, the train is in box_94, the mirror is in box_23, the disk is in box_02, the toaster is in box_42, and the lamp is in box_48. Respond in one word, only the answer and nothing else: Which box is the ice in? Answer:"

target = " " + "15" # התשובה שהסקריפט מחפש כרגע



print("\n" + "="*50)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

target_tokens = tokenizer(target, add_special_tokens=False).input_ids



print(f"Target string: '{target}'")

print(f"How Tokenizer sees the target: {[tokenizer.decode([t]) for t in target_tokens]} (IDs: {target_tokens})")

print(f"Last 3 tokens of the prompt: {[tokenizer.decode([t]) for t in inputs.input_ids[0, -3:]]}")



print("\nRunning model...")

with torch.no_grad():

    logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits[0, -1, :], dim=-1)

    top5 = probs.topk(5)



print("\nTop 5 predictions from the model:")

for i in range(5):

    tid = top5.indices[i].item()

    print(f"  {i+1}. Token: '{tokenizer.decode([tid])}' | Prob: {top5.values[i].item():.4%}")

print("="*50)
