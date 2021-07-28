import yaml

def parse_args(args):
    out = ""
    for arg in args:
        out += "\t" + arg + " (" + args[arg]["type"] + "): " + args[arg]["description"] + "\n"
    return out
    

def run():
    with open("sail_docs.yaml", 'r') as stream:
        try:
            sail_all_docs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    sail_docs = sail_all_docs["sail"]

    file = "import sail\nfrom sail import add_docstring\n\n"

    for fcn in sail_docs:
        docs = sail_docs[fcn]
        docstring = ""
        docstring += docs["header"] + "\n"
        docstring += docs["description"] + "\n\n"
        if "more_description" in docs:
            docstring += docs["more_description"] + "\n\n"
        if "math" in docs:
            docstring += ".. math::\n\t" + docs["math"] + "\n\n"

        if "notes" in docs:
            docstring += ".. note::\n\t" + docs["notes"] + "\n\n"
            
        docstring += "Args:\n" + parse_args(docs["args"]) + "\n"


        docstring += "Examples:\n\t" + docs["examples"].replace("\n", "\n\t")
        # add_docstring(getattr(sail, fcn), docstring)

        file += "\ndescr = r\"\"\"\n" + docstring + "\"\"\"\nadd_docstring(sail.%s, descr)\n" % fcn


    with open("_sail_docs.py", "w") as f:
        f.write(file)

if __name__ == "__main__":
    run()