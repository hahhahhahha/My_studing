"""Six analytic synthetic fixtures; not visual/QLA experimental evidence."""
import json
import math
from fifo_execution import FIFOExecution, Request


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-12), (a, b)


def run():
    checks = []
    e = FIFOExecution(1, 32)
    e.submit(Request('a', 0., .02, 0, .03, 100))
    e.advance(.1)
    close(e.results()['latency_seconds'][0], .05)
    checks.append('one task: upload plus service')

    e = FIFOExecution(1, 32)
    e.submit(Request('a', 0., .02, 0, .03, 100))
    e.submit(Request('b', 0., .03, 0, .04, 100))
    e.advance(.2)
    close(e.records['b']['started'], .05)
    close(e.records['b']['completed'], .09)
    checks.append('two uploads: FIFO analytic wait')

    e = FIFOExecution(1, 32)
    e.submit(Request('a', 0., .23, 0, .03, 100))
    for t in [.05, .1, .15, .2]:
        e.advance(t)
        assert e.records['a']['status'] == 'uploading'
    e.advance(.3)
    close(e.records['a']['completed'], .26)
    checks.append('upload spanning decision slots: no early admission')

    e = FIFOExecution(1, 32)
    e.submit(Request('a', 0., 0., 0, .23, 100))
    for t in [.05, .1, .15, .2]:
        e.advance(t)
        close(e.records['a']['started'], 0.)
        assert e.records['a']['status'] == 'running'
    e.advance(.25)
    close(e.records['a']['completed'], .23)
    checks.append('continuous service across broadcast/decision boundaries')

    e = FIFOExecution(1, 1)
    for request in [Request('a', 0., 0., 0, 1., 100), Request('b', 0., .1, 0, 1., 100), Request('c', 0., .2, 0, 1., 100)]:
        e.submit(request)
    e.advance(.3)
    assert e.records['c']['status'] == 'dropped'
    assert e.results()['generated'] == 3 and e.results()['accepted'] == 2
    e.advance(3.)
    assert e.results()['completed'] == 2 and e.results()['dropped'] == 1
    checks.append('whole-task finite buffer drop and count conservation')

    e = FIFOExecution(1, 32)
    e.submit(Request('a', 0., .2, 0, .3, 100))
    e.submit(Request('b', 0., .6, 0, .4, 100))
    e.advance(.1)
    assert e.results()['unfinished'] == 2 and e.results()['completed'] == 0
    e.advance(.7)
    assert e.results()['unfinished'] == 1
    e.advance(1.1)
    assert e.results()['unfinished'] == 0 and e.results()['completed'] == 2
    close(e.records['b']['completed'], 1.)
    checks.append('bounded drain: complete versus unfinished is explicit')
    print(json.dumps(dict(passed=True, fixtures=checks, count=len(checks), evidence='synthetic semantic tests only'), ensure_ascii=False))


if __name__ == '__main__':
    run()
