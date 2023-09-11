from os import mkdir, listdir, remove
from time import time
from math import ceil, floor
from random import sample
from defusedxml.ElementTree import parse
from bpmnparser import tasksFromFile


in_dir = "./BS-11722/"
out_dir = "./all_outputs/"
opt_dir = "./all_options/"
glob_count, count = 0, 0
print_warnings = False

all_processes = []
train, test, validation = [], [], []
split_to_samples = False

add_bpmn_filename = False
count_method = False
all_methods = set()
method_name = "url"
delete_elem = True
elem_to_cut = "?"

bag_of_words = set()
print_BoW = False

unique_fields = { "url" : {}, "method" : {}, "taskType" : { None : 0 } }

uniform_parameters = (("name", "sourceRef", "targetRef"), ("incoming", "outgoing"), ("url", "method"))
attrs_to_print = ("name", "taskType", "url", "method")
url_types = ("getAttribute", "getRefId", "setAttribute")

tasks_to_predict = { # Old
                    "serviceTask",
                    "userTask",
                    "scriptTask",
                    "exclusiveGateway",
                    "parallelGateway",
                    "sequenceFlow",
                    "endEvent"
                   }


def get_element_type(full_tag):
    return full_tag[ full_tag.rfind("}") + 1 : ]


def add_parameter_by_tag(element, tag_name):
    tmpElem = element.findall(".//{*}" + tag_name)
    return [ tmpElem[i].text for i in range(len(tmpElem)) ]


def add_parameter_by_attrib(element, attrib_name, parameter_name):
    tmpElem = element.find(".//*[@" + attrib_name + "='" + parameter_name + "']")

    if tmpElem != None:
        return { parameter_name : tmpElem.text }
    return { parameter_name : None }


def add_all_parameters(processTasksInfo, processTasksNeighbours, element, elementType):
    task_info = { "taskID" : len(processTasksInfo), "taskType" : elementType }

    for param_from_elem in uniform_parameters[0]:
        task_info.update({ param_from_elem : element.attrib.get(param_from_elem, None) })

    neighbours = tuple(add_parameter_by_tag(element, param_from_tag) for param_from_tag in uniform_parameters[1])
    if elementType == "sequenceFlow":
        neighbours = ([task_info["sourceRef"],], [task_info["targetRef"],])

    for param_from_attrib in uniform_parameters[2]:
        task_info.update(add_parameter_by_attrib(element, "name", param_from_attrib))

    processTasksInfo.update({ element.attrib["id"] : { attr : task_info.get(attr, None) for attr in attrs_to_print } })
    processTasksNeighbours.update({ element.attrib["id"] : neighbours })

    if elementType == "endEvent":
        return element.attrib["id"]
    return None


def get_xml_data(input: str):
    tasks_set = {
        "exclusiveGateway",
        "parallelGateway",
        "sequenceFlow",
        "callActivity",
        "startEvent",
        "endEvent",
        "intermediateThrowEvent",
        "serviceTask",
        "userTask",
        "scriptTask",
        "businessRuleTask",
        "subProcess"
    }

    tree = parse(input)
    root = tree.getroot()

    process = root.findall("{*}process")[0]
    processTasksInfo, processTasksNeighbours, endEvents = {}, {}, []

    for element in process.findall("./*"):
        if element.attrib.get("id", None) == None:
            continue
        element_type = get_element_type(element.tag)
        if element_type in tasks_set:
            endEventID = add_all_parameters(processTasksInfo, processTasksNeighbours, element, element_type)

            if endEventID != None:
                endEvents.append(endEventID)

    processBlocks = tasksFromFile(input)

    opt_cnt = 0
    ind = input.rfind('/')
    if ind != -1: input = input[ind + 1:]

    for key in processTasksInfo:
        outgoing = processTasksNeighbours[key][1]
        tmpdict = processTasksInfo[key]
        tmpurl = tmpdict["url"]
        if tmpurl is not None and (tmp := tmpurl.find('?')) > -1:
            tmpurl = tmpurl[:tmp]
        if unique_fields["url"].get(tmpurl, None) is None:
            unique_fields["url"][tmpurl] = len(unique_fields["url"])
        if unique_fields["taskType"].get(tmpdict["taskType"], None) is None:
            unique_fields["taskType"][tmpdict["taskType"]] = len(unique_fields["taskType"])
        if unique_fields["method"].get(tmpdict["method"], None) is None:
            unique_fields["method"][tmpdict["method"]] = len(unique_fields["method"])
        if len(outgoing) > 1 :#and processTasksInfo[key]["taskType"] not in ("exclusiveGateway", "parallelGateway"):
            opt_cnt += 1
            i = 0
            with open(opt_dir + f"{input}_{opt_cnt}.txt", 'w', encoding="utf8") as opt_file:
                opt_file.write("{\n    \"initTask\":\n\n" + processBlocks[key])
                for optID in outgoing:
                    if (task := processTasksInfo.get(optID, None)) == None:
                        continue
                    if task["taskType"] == "sequenceFlow":
                        if (task := processTasksNeighbours.get(optID, None)) == None:
                            continue
                        optID = task[1][0]
                    if (task := processBlocks.get(optID, None)) == None:
                        continue
                    i += 1
                    opt_file.write(f"\n\n    \"Option {i}\":\n\n{task}")
                opt_file.write("\n\n}")
            if i < 2: remove(opt_dir + f"{input}_{opt_cnt}.txt")

    return processTasksInfo, processTasksNeighbours, endEvents, processBlocks


def build_processes(processTasksInfo, processTasksNeighbours, endEvents, processBlocks, filename):
    def print_process(proc):
        global glob_count, count, out_dir, bag_of_words
        nonlocal processTasksInfo, processes

        curr_proc = []
        if add_bpmn_filename: curr_proc = [filename]
        curr_proc.extend([processTasksInfo[task][attr] for attr in attrs_to_print] for task in reversed(proc) if processTasksInfo[task]["taskType"] in tasks_to_predict)

        for elem in curr_proc[int(add_bpmn_filename):]:
            if count_method and elem[len(attrs_to_print) - 2] != None:             ##############################################
                if delete_elem:
                    if (newend := elem[len(attrs_to_print) - 2].find(elem_to_cut)) != -1:
                        elem[len(attrs_to_print) - 2] = elem[len(attrs_to_print) - 2][:newend]
                    for attr in url_types:
                        if (newend := elem[len(attrs_to_print) - 2].find(attr)) != -1:
                            elem[len(attrs_to_print) - 2] = elem[len(attrs_to_print) - 2][:newend + len(attr)]

                all_methods.add(elem[len(attrs_to_print) - 2])
            if elem[len(attrs_to_print) - 3] in tasks_to_predict:
                bag_of_words.add(tuple(elem))

        if curr_proc not in processes:
            processes.append(curr_proc)

            glob_count += 1
            count += 1

            out_file = open(out_dir + f"{filename}_{count}.txt", 'w', encoding="utf8")
            out_file.write(str(curr_proc) + '\n')
            out_file.close()

    def print_process2(proc):
        global glob_count, count, out_dir, bag_of_words
        nonlocal processTasksInfo, processes, processBlocks

        if proc not in processes:
            processes.append(proc)

            glob_count += 1
            count += 1

            with open(out_dir + f"{filename}_{count}.txt", 'w', encoding="utf8") as out_file:
                for elem in proc:
                    data = processTasksInfo[elem]
                    url = data["url"]
                    if url is not None and (tmp := url.find('?')) >= 0:
                        url = url[:tmp]
                    out_file.write(str(unique_fields["url"][url]) + ", " +
                                   str(unique_fields["taskType"][data["taskType"]]) + ", " +
                                   str(unique_fields["method"][data["method"]]) + '\n\n')

    def split_processes(processes):
        def round_length(curr_coef, model_coef):
            length = max(1, model_coef * count)
            if curr_coef < model_coef:
                length = ceil(length)
            elif curr_coef > model_coef:
                length = floor(length)
            else:
                length = round(length)

            return min(length, count - 2)


        global train, test, validation, glob_count, count

        train_coef = 0.8
        test_val_ratio = 0.5
        test_coef = (1. - train_coef) * test_val_ratio
        val_coef  = (1. - train_coef) * (1. - test_val_ratio)

        match count:
            case 0:
                lengths = [0, 0, 0]
            case 1 | 2:
                coefs = len(train) / glob_count / train_coef, \
                        len(test)  / glob_count /  test_coef, \
                        len(validation) / glob_count / val_coef

                if count == 1:
                    lengths = [int(elem == max(coefs)) for elem in coefs]

                    if sum(lengths) > 1:
                        chosen = sample([i for i in range(3) if lengths[i] != 0], k= 1)[0]
                        lengths = [int(i == chosen) for i in range(3)]
                elif count == 2:
                    lengths = [int(elem != min(coefs)) for elem in coefs]

                    if sum(lengths) < 2:
                        chosen = sample([i for i in range(3) if lengths[i] == 0], k= 1)[0]
                        lengths = [int(i != chosen) for i in range(3)]
            case 3:
                lengths = [1, 1, 1]
            case _:
                lengths = [round_length(len(train) / glob_count, train_coef),
                           round_length(len(test) / glob_count, test_coef),
                           round_length(len(validation) / glob_count, val_coef)]

                if sum(lengths) < count:
                    tmp_lengths = [0, 0, 0]
                    diff = count - sum(lengths)

                    for _ in range(diff):
                        coefs = (len(train) + lengths[0] + tmp_lengths[0]) / (glob_count - count + lengths[0] + tmp_lengths[0]) / train_coef, \
                                (len(test)  + lengths[1] + tmp_lengths[1]) / (glob_count - count + lengths[1] + tmp_lengths[1]) /  test_coef, \
                                (len(validation) + lengths[2] + tmp_lengths[2]) / (glob_count - count + lengths[2] + tmp_lengths[2]) / val_coef

                        tmp_lengths[[i for i in range(3) if coefs[i] == min(coefs)][0]] += 1

                    lengths = [length + tmp_len for length, tmp_len in zip(lengths, tmp_lengths)]
                elif sum(lengths) > count:
                    tmp_lengths = [0, 0, 0]
                    diff = sum(lengths) - count

                    for _ in range(diff):
                        coefs = (len(train) + lengths[0] + tmp_lengths[0]) / (glob_count + lengths[0] + tmp_lengths[0]) / train_coef, \
                                (len(test)  + lengths[1] + tmp_lengths[1]) / (glob_count + lengths[1] + tmp_lengths[1]) /  test_coef, \
                                (len(validation) + lengths[2] + tmp_lengths[2]) / (glob_count + lengths[2] + tmp_lengths[2]) / val_coef

                        ind = sorted(range(3), key = lambda x: coefs[x] - 3 * int((lengths[x] + tmp_lengths[x]) == 1), reverse= True)[0]
                        tmp_lengths[ind] -= 1
                    lengths = [length + tmp_len for length, tmp_len in zip(lengths, tmp_lengths)]

        train_sample = sample(range(count), k= lengths[0])
        test_sample = sample([i for i in range(count) if i not in train_sample], k= lengths[1])
        val_sample = sample([i for i in range(count) if (i not in train_sample) and (i not in test_sample)], k= lengths[2])

        train.extend(processes[i] for i in train_sample)
        test.extend(processes[i] for i in test_sample)
        validation.extend(processes[i] for i in val_sample)
        print(f"Train: {lengths[0]} / {count}, Test: {lengths[1]} / {count}, Validation: {lengths[2]} / {count}")
        print(f"Train: {lengths[0] / max(1, count) :.3f}, Test: {lengths[1] / max(1, count) :.3f}, Validation: {lengths[2] / max(1, count) :.3f}")


    processes = []
    procStack = [ [event] for event in endEvents ]

    while len(procStack) > 0:
        tmpProc = procStack.pop()
        parentTasks = processTasksNeighbours[tmpProc[-1]][0]

        for task in parentTasks:
            taskType = processTasksInfo.get(task, None)

            if (taskType == None) or (task in tmpProc):
                if taskType == None and print_warnings:
                    print(f"WARNING: element '{task}' is not found in processTasksInfo")
                continue

            if taskType["taskType"] != "startEvent":
                procStack.append(tmpProc + [task])
            else:
                print_process2(reversed(tmpProc + [task]))

    if split_to_samples: split_processes(processes)


if __name__ == "__main__":
    try:
        mkdir(out_dir)
    except FileExistsError:
        pass

    tz = time()
    for in_file in listdir(in_dir)[::-1]:
        count = 0
        print("File: ", in_file)
        ta = time()
        build_processes(*get_xml_data(in_dir + in_file), in_file[ : -5])
        print("Processes / Total processes:", count, "/", glob_count)
        print("Time: ", time() - ta, '\n')

    print("Total saved processes: ", glob_count)
    print(f"Train: {len(train)} / {glob_count}, Test: {len(test)} / {glob_count}, Validation: {len(validation)} / {glob_count}")
    if glob_count == 0: glob_count += 1
    print(f"Train: {len(train)  /  glob_count :.3f}, Test: {len(test)  /  glob_count :.3f}, Validation: {len(validation)  /  glob_count :.3f}")
    print("Total time: ", time() - tz)

    print(f"Lenghts: method = {len(unique_fields['method'])}, url = {len(unique_fields['url'])}, taskType = {len(unique_fields['taskType'])}")

    if print_BoW:
        out_file = open("bag_of_words.txt", 'w', encoding="utf-8")
        for elem in bag_of_words:
            out_file.write(str(elem) + '\n')
        out_file.close()

    with open("unique_params.txt", 'w', encoding='utf-8') as paramfile:
        paramfile.write("URLs:\n")
        params = list(unique_fields["url"].keys())
        params.sort(key= lambda x: unique_fields["url"][x])
        for elem in params:
            paramfile.write(str(elem) + "\n")
        params = list(unique_fields["taskType"].keys())
        params.sort(key= lambda x: unique_fields["taskType"][x])
        paramfile.write("Task types:\n")
        for elem in params:
            paramfile.write(str(elem) + '\n')
        params = list(unique_fields["method"].keys())
        params.sort(key= lambda x: unique_fields["method"][x])
        paramfile.write("Methods:\n")
        for elem in params:
            paramfile.write(str(elem) + '\n')