from os import listdir
import defusedxml.ElementTree as ET


PROCESSES = []
URL, METHOD, TASKTYPE = [], [], []
UNIQUE_FIELDS = { "url" : {}, "method" : {}, "taskType" : { None : 0 } }
SUBPROC_SKIPPABLE = ("association", "textAnnotation", "incoming", "outgoing")


##  Tokenize url, taskType or method
def FieldEncode(field_name: str, umtt: str) -> int:
    return UNIQUE_FIELDS[field_name][umtt]

##  Detokenize url, taskType or method
def FieldDecode(field_name: str, token: int) -> str:
    global URL, METHOD, TASKTYPE

    if field_name == "url":
        return URL[token]
    if field_name == "method":
        return METHOD[token]
    if field_name == "taskType":
        return TASKTYPE[token]


##  Encode: (URL, TaskType, Method) -> token
def TaskEncode(l_list: tuple[int, int, int]) -> int:
    global URL, TASKTYPE
    return l_list[0] + l_list[1] * len(URL) + l_list[2] * (len(URL) * len(TASKTYPE))

##  Decode: token -> (URL, TaskType, Method)
def TaskDecode(_num: int) -> tuple[int, int, int]:
    global URL, TASKTYPE

    url_id = _num % len(URL)
    met_id = _num // (len(TASKTYPE) * len(URL))
    tsk_id = (_num - url_id - met_id * (len(TASKTYPE) * len(URL))) // len(URL)
    return [url_id, tsk_id, met_id]



def proc_parse(proc):
    # Functions
    def get_element_type(full_tag):
        return full_tag[ full_tag.rfind("}") + 1 : ]

    def get_parameters_from_tag(element, tag_name):
        tmpElem = element.findall(".//{*}" + tag_name)
        return [ tmpElem[i].text for i in range(len(tmpElem)) ]

    def get_parameter_from_attrib(element, attrib_name, parameter_name):
        tmpElem = element.find(".//*[@" + attrib_name + "='" + parameter_name + "']")

        if tmpElem != None:
            return { parameter_name : tmpElem.text }
        return { parameter_name : None }

    # Delete after debugging
    def _checkChildTasks(task):
        for elem in task:
            print(get_element_type(elem.tag), elem.attrib.get("id", None))


    def get_task_info(task):
        _taskID = task.attrib.get("id", None)
        _task_type = get_element_type(task.tag)

        if (_taskID is None) or (_task_type in SUBPROC_SKIPPABLE):
            return

        _task_info = { "taskType" : _task_type }

        if _task_type == "subProcess":
            # _checkChildTasks(task)
            for _sp_task in task:
                get_task_info(_sp_task)

        _in  = get_parameters_from_tag(task, "incoming")
        _out = get_parameters_from_tag(task, "outgoing")
        if _task_type == "sequenceFlow":
            _in  = [task.attrib["sourceRef"],]
            _out = [task.attrib["targetRef"],]

        IN.update({ _taskID : _in })
        OUT.update({ _taskID : _out })

        for param_from_attrib in ("url", "method"):
            _task_info.update(get_parameter_from_attrib(task, "name", param_from_attrib))

        if _task_type == "startEvent":
            START_EVENTS.append(_taskID)
        if _task_type == "endEvent":
            END_EVENTS.append(_taskID)
        TASKS.update({ _taskID : _task_info })
    

    def encode_task_fields(task_info: dict) -> tuple[int, int, int]:
        def url_cleanup(_url):
            if (_url is not None) and (ind := _url.find('?')) > -1:
                _url = _url[:ind]
            return _url

        # change info in TASKS (url, method, taskType) from str to int
        # based on the size of corresponding UNIQUE_FIELDS dict
        # change proc_build or just rewrite taskID -> final (encoded) token
        global URL, TASKTYPE, METHOD, UNIQUE_FIELDS

        _tmp = url_cleanup(task_info["url"]), task_info["taskType"], task_info["method"]

        if (_url := UNIQUE_FIELDS["url"].get(_tmp[0], len(URL))) == len(URL):
            UNIQUE_FIELDS["url"].update({ _tmp[0] : _url })
            URL.append(_tmp[0])

        if (_ttype := UNIQUE_FIELDS["taskType"].get(_tmp[1], len(TASKTYPE))) == len(TASKTYPE):
            UNIQUE_FIELDS["taskType"].update({ _tmp[1] : _ttype })
            TASKTYPE.append(_tmp[1])

        if (_method := UNIQUE_FIELDS["method"].get(_tmp[2], len(METHOD))) == len(METHOD):
            UNIQUE_FIELDS["method"].update({ _tmp[2] : _method })
            METHOD.append(_tmp[2])

        return (_url, _ttype, _method)


    def encode_proc(proc):
        return proc
        for i in range(len(proc)):
            _tmp = TASKS[proc[i]]
            if type(_tmp) is dict:
                _tmp = encode_task_fields(_tmp)
                TASKS[proc[i]] = _tmp
            proc[i] = TaskEncode(_tmp)
        return proc


    def filter_seq_flow():
        for _task, _info in TASKS.items():
            if _info["taskType"] == "sequenceFlow":
                # by design checks if sequenceFlow has one in/out task:
                _inTask, _outTask = *IN[_task], *OUT[_task]

                _tmp = OUT[_inTask]
                _tmp[_tmp.index(_task)] = _outTask
                OUT[_inTask] = _tmp

                _tmp = IN[_outTask]
                _tmp[_tmp.index(_task)] = _inTask
                IN[_outTask] = _tmp

    def is_valid_process(proc):
        return (TASKS[proc[ 0]]["taskType"] == "startEvent") and\
               (TASKS[proc[-1]]["taskType"] ==   "endEvent")


    def build_proc():
        _proc_buffer = [ [s_event,] for s_event in START_EVENTS ]

        while len(_proc_buffer) != 0:
            _proc = _proc_buffer.pop()
            _nextTasks = OUT[_proc[-1]]

            if len(_nextTasks) == 0:
                if is_valid_process(_proc):
                    PROCS.append(encode_proc(_proc))
                continue

            for _next in _nextTasks:
                _afterNext = OUT[_next]

                if _next not in _proc:
                    if len(_afterNext) > 0:
                        _proc_buffer.append(_proc + [_next])
                    elif is_valid_process(_proc + [_next]):
                        PROCS.append(encode_proc(_proc + [_next]))
                    continue

                for _after in _afterNext:
                    if _after not in _proc:
                        _proc_buffer.append(_proc + [_next, _after])


    def encode_task_to3(taskID):
        if (_res := TASKS_ENC.get(taskID, None)) is None:
            _res = encode_task_fields(TASKS[taskID])
            TASKS_ENC[taskID] = _res
        return _res

    def encode_procs_to3():
        for proc in PROCS:
            PROCESSES.append(tuple((encode_task_to3(task) for task in proc)))


    # Code
    IN, OUT, TASKS, TASKS_ENC = {}, {}, {}, {}
    START_EVENTS, END_EVENTS, PROCS = [], [], []

    _a = time()
    for task in proc:
        get_task_info(task)
    _b = time()
    filter_seq_flow()
    _c = time()
    build_proc()
    _d = time()
    encode_procs_to3()
    print("t1:", _b - _a, ", t2:", _c - _b, ", t3:", _d - _c, ", t4:", time() - _d)


def build_processes(fpath: str) -> None:
    root = ET.parse(fpath).getroot()

    for proc in root.findall("{*}process"):
        proc_parse(proc)

def encode_all_procs():
    global PROCESSES

    PROC_SET = set()
    while len(PROCESSES):
        PROC_SET.add(tuple((TaskEncode(task) for task in PROCESSES.pop())))
    PROCESSES = PROC_SET
    # for i in range(len(PROCESSES)):
    #     PROCESSES[i] = tuple((TaskEncode(task) for task in PROCESSES[i]))



if __name__ == "__main__":
    from time import time

    t1 = time()
    l = listdir("./BS-11722")
    for i, _file in enumerate(l):
        t2 = time()
        build_processes("./BS-11722/" + _file)
        print("Done", i, "; Time =", time() - t2)
    t_b = time() - t1
    print("Encoding processes...")
    encode_all_procs()
    t_a = time() - t1
    print(len(URL), len(METHOD), len(TASKTYPE))
    print("Encoding time:", t_a - t_b)
    print("Overall time:", t_a)
    print(len(PROCESSES))

    ## Print processes
    with open("processes.txt", 'w', encoding='utf-8') as paramfile:
        for proc in PROCESSES:
            paramfile.write(str(proc)[1:-1] + '\n')


    ## Print URLs, task types and methods
    with open("unique_params.txt", 'w', encoding='utf-8') as paramfile:
        paramfile.write("URLs:\n")
        params = list(UNIQUE_FIELDS["url"].keys())
        params.sort(key= lambda x: UNIQUE_FIELDS["url"][x])
        for elem in params:
            paramfile.write(str(elem) + "\n")
        params = list(UNIQUE_FIELDS["taskType"].keys())
        params.sort(key= lambda x: UNIQUE_FIELDS["taskType"][x])
        paramfile.write("Task types:\n")
        for elem in params:
            paramfile.write(str(elem) + '\n')
        params = list(UNIQUE_FIELDS["method"].keys())
        params.sort(key= lambda x: UNIQUE_FIELDS["method"][x])
        paramfile.write("Methods:\n")
        for elem in params:
            paramfile.write(str(elem) + '\n')