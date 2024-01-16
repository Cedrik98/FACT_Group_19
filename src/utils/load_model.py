import warnings
warnings.filterwarnings("ignore")

import textattack
import transformers



def load_model_and_tokenizer(model_name, max_length, device, valid_models):
    if model_name.replace('thaile/', '') not in valid_models:
        print("CAUTION! You are running a model not in the model cards.")

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        model = ORTModelForSequenceClassification.from_pretrained(model_name, 
                                                                  export=True, 
                                                                  provider="CUDAExecutionProvider", 
                                                                  use_io_binding=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except:  
        print(f"Error using Optimum Runtime, using default model settings")
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name) 
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    if max_length:
        tokenizer.model_max_length = max_length
    
    model.to(device)
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    return model, tokenizer, model_wrapper
