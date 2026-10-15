# Alibaba 2017 acquisition status

2026-10-09: no archive downloaded, no capacity file generated, no archive checksum verified.

The official README links to:

http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2017Traces/alibaba-trace-2017.tar.gz

A read-only HEAD probe against the same host via HTTPS failed in both the default and elevated execution context:

`curl: (6) Could not resolve host: aliopentrace.oss-cn-beijing.aliyuncs.com`

One elevated invocation also had a missing executable search path; repeating with `C:/Windows/System32/curl.exe` removed that path issue but DNS resolution still failed.

This does not demonstrate that the official public download is unavailable to the user. It demonstrates that acquisition has not succeeded in this task's current execution context. No substitute data were used.

Raw archive and extracted `server_usage.csv` must be checked against the official release/schema before model testing. The official checksum names `trace_201708.tgz`; archive naming/content correspondence remains to be verified after successful acquisition.
