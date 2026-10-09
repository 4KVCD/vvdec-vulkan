# gen_parse_contexts.py: writes source/Lib/VulkanLib/shaders/parsectx.slang, the first context of each of vvdec's
# context sets (CommonLib/Contexts.cpp, ContextSetCfg::addCtxSet in declaration order) and their count, as #defines that
# the GPU's parser (shaders/parse.slang) and the host (VulkanLib/GpuParse.cpp, which checks them against Ctx's) include.
#
#   python scripts/gen_parse_contexts.py
import os
import re

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
src = os.environ.get('VVDEC_ROOT', root)
source = open(os.path.join(src, 'source/Lib/CommonLib/Contexts.cpp'), encoding='utf-8').read()

# the declarations, each up to the next one: "const CtxSet ContextSetCfg::Name =" or "...::Name[] ="
decl = re.compile(r'const CtxSet\s+ContextSetCfg::(\w+)\s*(\[\])?\s*=')
matches = list(decl.finditer(source))
offset = 0
lines = []
for i, m in enumerate(matches):
    name, array = m.group(1), m.group(2)
    end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
    body = source[m.end():end]
    sets = [s.end() for s in re.finditer(r'addCtxSet\s*\(\s*\{', body)]
    if not sets:
        continue   # a combined set (Sao, Alf)
    for k, start in enumerate(sets):
        row = re.match(r'\s*\{([^}]*)\}', body[start:])
        size = len([v for v in row.group(1).split(',') if v.strip()])
        label = name + ('_%d' % k if array else '')
        lines.append('#define CTX_%s %d   // %d' % (label, offset, size))
        offset += size

out = ['// The first context of each of vvdec\'s context sets (CommonLib/Contexts.cpp, in the order ContextSetCfg::addCtxSet',
       '// makes them; a set\'s size after it) and their count: written by scripts/gen_parse_contexts.py, included by the',
       '// GPU\'s parser (parse.slang) and the host (GpuParse.cpp checks them against Ctx\'s).',
       ''] + lines + ['', '#define CTX_COUNT %d' % offset, '']
open(os.path.join(root, 'source/Lib/VulkanLib/shaders/parsectx.slang'), 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
print('%d contexts' % offset)
