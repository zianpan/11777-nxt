


class PromptGenerator:

    def __init__(self,choices_map):
        # self.model = model
        # self.model.eval()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        # for key, value in kargs.items():
        #     setattr(self, key, value)
        
        self.choices_map = choices_map
        pass
    
    def generate_template(self):
        pass

    def generate_one_shot(self):
        pass

    def generate_question(self):
        pass

class PromptGeneratorABCD(PromptGenerator):
    def __init__(self,choices_map = {0:'A', 1:'B', 2:'C', 3:'D'}, **kargs):
        # super().__init__(choices_map)
        self.choices_map = choices_map
        for key, value in kargs.items():
            setattr(self, key, value)
        
    
    def generate_one_shot(self,question, choices, rationale, base_ans):
        prompt = f"Question: {question}\nChoices: {self.choices_map[0]}. {choices[0]} {self.choices_map[1]}. {choices[1]} {self.choices_map[2]}. {choices[2]} {self.choices_map[3]}. {choices[3]}\nRationale: {{{''.join(rationale)}}}\nAnswer: {{{self.choices_map[base_ans]}}}"
        return prompt

    def generate_question(self,question, choices):
        prompt = f"Question: {question}\nChoices: {self.choices_map[0]}. {choices[0]} {self.choices_map[1]}. {choices[1]} {self.choices_map[2]}. {choices[2]} {self.choices_map[3]}. {choices[3]}\nRationale: {{FILL IN Rationale}} Answer: {{FILL IN Answer}}"
        return prompt
    
    def generate_template(self,examples: list):
        prompt_list = []
        for example in examples:
            question = example["question"]
            choices = example["choices"]
            rationale = example['rationales']
            direct_ans = example['direct_answers']
            base_ans = example["correct_choice_idx"]
            prompt = self.generate_one_shot(question, choices, rationale, base_ans)
            prompt_list.append(prompt)
        
        if getattr(self, 'prompt_template', None):
            prompt_list = self.prompt_template.format('\n'.join(prompt_list))
            return prompt_list
        
        return '\n\n'.join(prompt_list)

class PromptGenerator0123(PromptGeneratorABCD):
    def __init__(self,choices_map = {0:0, 1:1, 2:2, 3:3}, **kargs):
        super().__init__(**kargs)
        self.choices_map = choices_map

        


        
    # def base_oneshot_generator(self,question, choices, rationale, direct_ans, base_ans):
    #     prompt = f"Question: {question}\nChoices: A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]}\nRationale: {{{''.join(rationale)}}}\nAnswer: {{{toABCD[base_ans]}}}"
    #     return prompt

    # def base_fewshot_generator(self,val_aokvqa,coco_id_filename, num_shots = 3):
    #     prompt_list = []
    #     sample_used = set()
    #     for i in range(num_shots):
    #         meta_data_one_sample = val_aokvqa[i]
    #     # meta_data_one_sample
    #         # TODO modify base_path
    #         base_path = "/home/ubuntu/data/coco/val2017/"
    #         img_id = meta_data_one_sample["image_id"]
    #         sample_used.add(img_id)
    #         img_file = coco_id_filename[img_id]
    #         img_path = base_path + img_file  
    #         base_ans = meta_data_one_sample["correct_choice_idx"]
    #         rationale =  meta_data_one_sample['rationales']
    #         direct_ans = meta_data_one_sample['direct_answers']
    #         toABCD = {0:'A', 1:'B', 2:'C', 3:'D'}

    #         question = meta_data_one_sample["question"]
    #         choices = meta_data_one_sample["choices"]
    #         prompt = self.base_oneshot_generator(question, choices, rationale, direct_ans, base_ans)
    #         prompt_list.append(prompt)
            
    #     return '\n'.join(prompt_list), sample_used


    # def question_generator(self,question, choices):
    #     prompt = f"Question: {question}\nChoices: A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]}\nRationale: {{FILL IN Rationale}} Answer: {{FILL IN Answer}}"
    #     return prompt