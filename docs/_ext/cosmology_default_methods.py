from docutils import nodes
from docutils.parsers.rst import Directive

from cosmology.background import Cosmology


class CosmologyDefaultMethods(Directive):

    def run(self):
        table = nodes.table(cols=3, width='100%')
        group = nodes.tgroup()
        head = nodes.thead()
        body = nodes.tbody()

        table += group
        group += nodes.colspec()
        group += nodes.colspec()
        group += nodes.colspec()
        group += head
        group += body

        row = nodes.row()
        row += nodes.entry('', nodes.paragraph('', nodes.Text('Method')))
        row += nodes.entry('', nodes.paragraph('', nodes.Text('Requires')))
        row += nodes.entry('', nodes.paragraph('', nodes.Text('Comment')))
        head += row

        defaults = Cosmology._default_methods()

        for method, requires, default in defaults:
            requires = ', '.join(requires)
            comment = default.__doc__ or ''
            row = nodes.row()
            row += nodes.entry('', nodes.paragraph('', nodes.Text(method)))
            row += nodes.entry('', nodes.paragraph('', nodes.Text(requires)))
            row += nodes.entry('', nodes.paragraph('', nodes.Text(comment)))
            body += row

        return [table]


def setup(app):
    app.add_directive('cosmology-default-methods', CosmologyDefaultMethods)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
