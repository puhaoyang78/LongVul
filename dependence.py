"""Extract native Joern dependence edges from individual C/C++ source files."""
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import gzip
from itertools import accumulate
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

EDGE_TYPES = ("REACHING_DEF", "CDG")
JOERN = Path(os.environ.get("JOERN_DIR", "/home/phy/joern"))
JAVA = Path(os.environ.get("JAVA_HOME", "/home/phy/jdk21"))
BATCH_FUNCTIONS = 128
BATCH_BYTES = 1_000_000
ISOLATE_BYTES = 128 * 1024
TIMEOUT_SECONDS = 180
ISOLATED_TIMEOUT_SECONDS = 420
MAX_DEFINITIONS = 4000
FRONTEND_ARGS = ("--log-problems", "--log-preprocessor", "--enable-file-content")
CPP_NODES = {"namespace_definition", "template_declaration", "class_specifier", "qualified_identifier",
             "reference_declarator", "operator_name", "destructor_name", "linkage_specification",
             "lambda_expression", "raw_string_literal", "new_expression", "delete_expression", "try_statement",
             "placeholder_type_specifier", "concept_definition", "structured_binding_declarator"}
SOURCE_CODE_TYPES = {"CALL", "IDENTIFIER", "LITERAL", "LOCAL", "METHOD_PARAMETER_IN", "RETURN", "CONTROL_STRUCTURE",
                     "FIELD_IDENTIFIER", "TYPE_REF", "METHOD_REF", "JUMP_TARGET", "UNKNOWN"}


def joern_version():
    jars = list((JOERN / "lib").glob("io.joern.joern-cli-*.jar"))
    for binary in (JOERN / "joern-parse", JOERN / "joern-export", JAVA / "bin/java"):
        if not binary.is_file():
            raise FileNotFoundError(binary)
    if len(jars) != 1:
        raise ValueError(f"Expected one joern-cli JAR under {JOERN}")
    return jars[0].stem.removeprefix("io.joern.joern-cli-")


def run_joern(command, directory, name, *, timeout=TIMEOUT_SECONDS):
    env = dict(os.environ, JAVA_HOME=str(JAVA), PATH=f"{JAVA / 'bin'}:{os.environ['PATH']}",
               JAVA_OPTS="-Xmx6g -Xms256m -XX:ActiveProcessorCount=4",
               JAVA_TOOL_OPTIONS="-Xmx6g -XX:ActiveProcessorCount=4", SL_LOGGING_LEVEL="INFO")
    start = perf_counter()
    logfile, rss = directory / f"{name}.log", directory / f"{name}.rss"
    with logfile.open("w") as log:
        process = subprocess.Popen(["/usr/bin/time", "-f", "%M", "-o", str(rss), *map(str, command)],
                                   env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=8)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    stats = dict(seconds=perf_counter()-start, peak_rss_kib=int(rss.read_text().splitlines()[-1]), exit_code=code)
    if code:
        raise RuntimeError(f"Joern {name} exited {code}. {logfile.read_text()[-3000:]}")
    return stats


def language_suffix(source, parser, filename=None):
    extension = Path(filename).suffix if filename else ""
    if extension == ".c":
        return ".c"
    if extension in {".C", ".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hh", ".hxx"}:
        return ".cpp"
    nodes = [parser.parse(source.encode()).root_node]
    while nodes:
        node = nodes.pop()
        if node.type in CPP_NODES and not node.has_error:
            return ".cpp"
        nodes.extend(node.named_children)
    return ".c"


def source_position(source, starts, line, column):
    if not (0 < line < len(starts) and column > 0):
        return None
    lo, hi = starts[line-1:line+1]
    row = source[lo:hi]
    if row.isascii():
        return lo+column-1 if column <= len(row) else None
    units = list(accumulate((1 + (ord(c) > 0xFFFF) for c in row), initial=0))
    return lo+bisect_right(units, column-1)-1 if column <= units[-1] else None


def physical_lines(source):
    return [0, *(m.end() for m in re.finditer(r"\r\n|\r|\n", source) if m.end() < len(source)), len(source)]


def parser_messages(log, filenames):
    facts = {name: Counter() for name in filenames}
    problem_file = re.compile(r"\bFile:\s+'([^']+)'")
    topic = None
    for row in log.splitlines():
        if re.match(r"\d{4}-\d{2}-\d{2} ", row):
            topic = None
        if "Parse problem '" in row and "occurred!" in row:
            topic = "parse_problems"
        elif "Preprocessor statement '" in row and "found!" in row:
            topic = "preprocessor_statements"
        if line := re.search(r"Parsed '([^']+)'", row):
            file = Path(line[1]).name
            if file in facts:
                facts[file]["cdt_parsed"] += 1
        if topic and (line := problem_file.search(row)) and Path(line[1]).name in facts:
            facts[Path(line[1]).name][topic] += 1
            topic = None
        if line := re.search(r"Failed to generate a CPG for: '([^']+)'", row):
            file = Path(line[1]).name
            if file in facts:
                facts[file]["ast_creation_failed"] += 1
    return facts


def read_native(directory, sources, log=""):
    csv.field_size_limit(sys.maxsize)
    coordinates, methods, files, overlays = {}, defaultdict(list), set(), set()
    for header in directory.glob("nodes_*_header.csv"):
        with header.open(newline="") as handle:
            fields = next(csv.reader(handle))
        if fields[:2] != [":ID", ":LABEL"]:
            raise ValueError(f"Unexpected Joern node header {header.name}")
        columns = {key.partition(":")[0]: i for i, key in enumerate(fields) if i >= 2}
        with header.with_name(header.name.replace("_header", "_data")).open(newline="") as handle:
            for row in csv.reader(handle):
                if len(row) != len(fields):
                    raise ValueError(f"Misaligned Joern node columns in {header.name}")
                node, kind = row[:2]
                line = int(row[columns["LINE_NUMBER"]] or 0) if "LINE_NUMBER" in columns else 0
                col = int(row[columns["COLUMN_NUMBER"]] or 0) if "COLUMN_NUMBER" in columns else 0
                code = row[columns["CODE"]].replace("\\\\", "\\") if "CODE" in columns else ""
                name = row[columns["NAME"]] if "NAME" in columns else ""
                coordinates[node] = (kind, line, col, code if kind == "METHOD" else code[:256], name)
                if kind == "FILE" and Path(name).name in sources:
                    files.add(Path(name).name)
                if kind == "META_DATA":
                    overlays.update(row[columns["OVERLAYS"]].split(";"))
                if kind == "METHOD" and row[columns["IS_EXTERNAL"]] != "true" and row[columns["NAME"]] != "<global>":
                    file = Path(row[columns["FILENAME"]]).name
                    if file in sources:
                        methods[file].append((node, *(int(row[columns[k]]) if row[columns[k]] else None for k in
                            ("LINE_NUMBER_END", "COLUMN_NUMBER_END", "OFFSET", "OFFSET_END"))))
    if not {"base", "controlflow", "dataflowOss"}.issubset(overlays):
        raise ValueError(f"Joern export is missing required overlays: {sorted(overlays)}")

    def edges(kind):
        path = directory / f"edges_{kind}_data.csv"
        if path.is_file():
            with path.open(newline="") as handle:
                for left, right, native, *_ in csv.reader(handle):
                    if native != kind or left not in coordinates or right not in coordinates:
                        raise ValueError(f"Malformed {kind} edge in native export")
                    yield left, right

    ast, parents, native = defaultdict(list), {}, defaultdict(list)
    for parent, child in edges("AST"):
        ast[parent].append(child)
        parents[child] = parent
    for kind in (*EDGE_TYPES, "CFG"):
        for left, right in edges(kind):
            native[left].append((kind, right))

    output = []
    messages = parser_messages(log, sources)
    for file, source in sources.items():
        raw, endpoints, locations = [], set(), []
        counts, structure = Counter(), Counter()
        file_nodes = set()
        starts = physical_lines(source)
        for method, end_line, end_col, offset, offset_end in methods[file]:
            _, line, col, code, name = coordinates[method]
            if not line:
                counts["methods_missing_line"] += 1
                continue
            point = source_position(source, starts, line, col)
            # Joern caps long CODE properties, so verify the coordinate using its unmodified prefix.
            prefix = code.rstrip("\r\n")[:96]
            if (point is None or not prefix or not (line <= (end_line or line) < len(starts))
                    or not source[point:point+2*len(prefix)].replace("\r\n", "\n").startswith(prefix)):
                counts["methods_position_mismatch"] += 1
                continue
            locations.append(dict(file=file, line=line, column=col, end_line=end_line, end_column=end_col,
                                  offset=offset, offset_end=offset_end, name=name))
            owned, pending = set(), [method]
            while pending:
                node = pending.pop()
                if node in owned or (node != method and coordinates[node][0] == "METHOD"):
                    continue
                owned.add(node)
                pending.extend(ast[node])
            file_nodes.update(owned)
            structure.update(coordinates[node][0] for node in owned)
            if any(kind == "CFG" and right in owned for kind,right in native[method]):
                counts["cfg_entries"] += 1
            if any(coordinates[node][0] == "METHOD_RETURN" for node in owned):
                counts["cfg_return_nodes"] += 1
            for left in sorted(owned):
                for kind, right in native[left]:
                    if right in owned:
                        counts[kind] += 1
                        if kind in EDGE_TYPES:
                            endpoints.update((left, right))
                            raw.append((kind, left, right))
        nodes = set(endpoints)
        for node in sorted(endpoints):
            while (parent := parents.get(node)) in file_nodes and coordinates[parent][0] in SOURCE_CODE_TYPES:
                nodes.add(parent)
                node = parent
        if not locations:
            status = "method_position_mismatch" if methods[file] else "source_method_missing"
        elif not counts["CFG"]:
            status = "cfg_missing"
        elif counts["cfg_entries"] != len(locations) or counts["cfg_return_nodes"] != len(locations):
            status = "cfg_partial"
        else:
            status = "ok"
        counts.update(messages[file])
        output.append(dict(status=status, methods=len(locations), file_in_cpg=file in files,
                           method_locations=locations, joern_nodes=len(file_nodes), endpoint_nodes=len(endpoints),
                           nodes={node: (*coordinates[node][:3], coordinates[node][3][:256],
                                         coordinates[node][4], parents.get(node)) for node in sorted(nodes)},
                           edges=raw, counts=dict(counts), structure=dict(structure)))
    return output


def extract(sources, filenames=None):
    """Parse original source only. A failed batch has an explicit status and no graph."""
    from tree_sitter import Language, Parser
    from tree_sitter_cpp import language

    parser = Parser(Language(language()))
    with TemporaryDirectory(prefix="longvul-joern-") as folder:
        work = Path(folder)
        source_dir = work / "src"
        source_dir.mkdir()
        staged = {}
        for i, source in enumerate(sources):
            raw = source.encode()
            suffix = language_suffix(source, parser, filenames[i] if filenames else None)
            filename = f"{i:04d}{suffix}"
            (source_dir / filename).write_bytes(raw)
            staged[filename] = source
        cpg, export = work / "cpg.bin", work / "export"
        timings = {}
        started = perf_counter()
        try:
            timeout = ISOLATED_TIMEOUT_SECONDS if len(sources) == 1 and len(sources[0].encode()) > ISOLATE_BYTES else TIMEOUT_SECONDS
            timings["parse"] = run_joern([JOERN / "joern-parse", source_dir, "--output", cpg, "--language", "NEWC",
                                          "--max-num-def", MAX_DEFINITIONS, "--frontend-args", *FRONTEND_ARGS],
                                         work, "parse", timeout=timeout)
            timings["export"] = run_joern([JOERN / "joern-export", "--repr", "all", "--format", "neo4jcsv",
                                            "--out", export, cpg], work, "export")
            before = perf_counter()
            log = (work / "parse.log").read_text()
            graphs = read_native(export, staged, log)
            timings["ddg_definition_limit_messages"] = len(re.findall(r" has more than \d+ definitions", log))
            timings["decode_seconds"] = perf_counter()-before
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            timings["failure"] = f"{type(error).__name__}: {error}"
            phase = "export" if "parse" in timings else "build"
            graphs = [dict(status=f"cpg_{phase}_failed", methods=0, file_in_cpg=False, method_locations=[],
                           joern_nodes=0, endpoint_nodes=0, nodes={}, edges=[], counts={}, structure={})
                      for _ in sources]
        for graph, filename in zip(graphs, staged, strict=True):
            graph["language"] = "C++" if filename.endswith(".cpp") else "C"
        timings["seconds"] = perf_counter()-started
        return graphs, timings


def batch_rows(rows):
    batch, size = [], 0
    for row in rows:
        added = len(row["func"].encode())
        if added > ISOLATE_BYTES:
            if batch:
                yield batch
                batch, size = [], 0
            yield [row]
            continue
        if batch and (len(batch) >= BATCH_FUNCTIONS or size + added > BATCH_BYTES):
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += added
    if batch:
        yield batch


def cached_graphs(rows, directory, *, prepare=False):
    """Yield batch graphs in release order. Completed batches are committed atomically."""
    version = joern_version()
    if prepare:
        directory.mkdir(parents=True, exist_ok=True)
    for i, batch in enumerate(batch_rows(rows)):
        path = directory / f"{i:04d}.json.gz"
        members = [[r["sample_key"], r["idx"], len(r["func"])] for r in batch]
        if not path.exists():
            if not prepare:
                raise FileNotFoundError(f"Missing Joern batch {path}")
            graphs, times = extract([r["func"] for r in batch])
            payload = dict(version=version, members=members, graphs=graphs, cost=times)
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wt") as handle:
                json.dump(payload, handle, separators=(",", ":"))
            tmp.replace(path)
            print(directory.name, i, dict(Counter(graph["status"] for graph in graphs)),
                  round(times["seconds"], 2), flush=True)
        else:
            with gzip.open(path, "rt") as handle:
                payload = json.load(handle)
        if payload["members"] != members or payload["version"] != version or len(payload["graphs"]) != len(batch):
            raise ValueError(f"Invalid Joern cache membership in {path}")
        yield payload["graphs"], payload["cost"]
