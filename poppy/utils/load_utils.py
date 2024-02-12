import pickle


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "poppy.trainer_pop":
            renamed_module = "poppy.trainers.trainer_base"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def robust_load(file_obj):
    return RenameUnpickler(file_obj).load()
