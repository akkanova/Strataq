# Engine requests and command replies

An engine advertises `requests` on its `READY` line. Python then uses this
protocol for every generation, including when only one request is running.
There is no solo request, client-selected execution slot, admission handoff,
or promotion between generation modes.

```text
RUN <request-id> <max-new> [key=value ...] <token,token,...>
RUNI <request-id> <max-new> [key=value ...] <picture-file> <token,token,...>
CANCEL <request-id>
ASK <command-id> <command>
```

IDs contain letters, digits, underscores and hyphens. Python assigns IDs per
engine process. A generation keeps its identity if it resumes after preemption.
Sampling options and picture files have the same meanings as before.

Generation events are complete lines prefixed with `R <request-id>`:

```text
R r1 STARTED
R r1 PP <read> <total> <milliseconds> <tokens-per-second>
R r1 T <token-id>
R r1 LP <logprob> [<token-id>:<logprob> ...]
R r1 DONE <produced> <prompt-tokens> <prompt-ms> <decode-ms> <finish> [statistics ...]
R r1 ERR <reason>
```

A request with `logprobs=K` (and optionally `allowed=<id,id,...>`) gets an `LP` line right after each `T` line:
the token's log probability, then the K most likely tokens' (or the allowed ids'), from the logits before
penalties and temperature.

`STARTED` means the engine has admitted the request. Queue deadlines remain
active until then. `DONE` supplies the engine's existing statistics, including
its optional model-specific tail. A `preempted` completion preserves output:
Python resubmits the same identity with the accepted output appended to its
prompt and the remaining output allowance. It combines the segment statistics.
Cancellation targets that identity, including while it waits for admission.
An unknown or already-completed cancellation has no effect.

Every line of an ASK reply is prefixed with `C <command-id>`. The command's
existing end marker still ends the reply:

```text
ASK c1 CHANGE_SETTINGS rows=4
C c1 MEMORY ...
C c1 SETTINGS rows=4 ...
C c1 END_MEMORY
```

Python routes each identity to its own queue. A generation does not hold a
command lock. A timed-out command's late reply is discarded, so it cannot be
mistaken for a later command's answer. Process exit wakes all queues; requests
from a previous process cannot write into its successor.

The paged engine chooses execution placement and admission priority. Its
`request_capacity` is a scheduling capacity, not a set of empty conversations.
`request_queue=1` tells Python to let the engine queue and preempt requests.
Conversation state and working buffers are created on demand. No extra state
is reserved for a distinguished request. Cached contexts have independent
`conversation<N>` addresses for inspection.

The paged command loop runs between decode steps, so settings can change while
one or several requests are generating. Commands still wait for a model
operation that is in progress. During prefill, the paged engine services
`LIST_CONVERSATIONS`, `GET_CONTEXT_WINDOWS` and `GET_MEMORY_USE` between prompt
chunks, without admitting queued generation or settings changes. A reading
conversation exposes the prompt tokens processed so far. Context reads allow
up to five minutes for a reply, since an individual GPU chunk can exceed the
ordinary query deadline. Command identities keep late replies isolated if an
operation still outlasts its caller's timeout.

Queued settings make prefill yield at its next completed chunk. The engine
returns `DONE ... preempted`, retaining the processed prefix, and handles the
settings through its ordinary command loop. Python resubmits the same request
identity and combines its statistics, as with other scheduler preemptions.

The older Qwen3.8, Qwen3.6, DeepSeek and SYCL handlers use the shared transport
adapter. It chooses their internal placement, frames their events, and combines
their admission and completion statistics into one DONE. Their `GEN`, `BGEN`,
`BADM`, `BT` and `BDONE` details stay inside the adapter. Sequential engines can
still finish an operation before servicing another command; this protocol does
not make an old model implementation concurrent or add live settings support.

## Live statistics

The paged engine writes one untagged `STATS` line a second, unasked, from its
own thread, so the lines keep coming during a long prompt chunk. The rest of
the line is one JSON object:

```text
STATS {"seconds":12.004,"gpus":[{"id":"gpu0","to_gpu":81920,"from_gpu":4096,"experts":12}],"decode":{"steps":310,"rows":806,"conversations":620,"routed_rows":806,"routings":425568,"experts":382900,"cpu_experts":0,"guesses":496,"kept":278,"guess_room":1240},"prefill":{"tokens":40960,"ms":38211.500}}
```

`seconds` is the time since the engine became ready. Each of `gpus` is one card
(`id` as in `GET_MEMORY_USE`). `to_gpu` and `from_gpu` are the bytes that moved
from host memory to the card and back since the engine started: copies, and the card's reads
and writes of host memory in place. `experts` is how many experts were among the
bytes sent to the card; an expert sent again counts again, and one split over
several cards counts once, on the card that gets its first part. Older engines
leave it out. They are totals, so the difference between
two lines over the difference of their `seconds` is the rate, even when a line
is late.

`decode` and `prefill` are totals too. `decode.steps` counts the decode steps
(one pass through the model each) and `rows` the rows they ran: a row is one
token of a conversation, or one guessed token checked. `conversations` counts the
conversations each completed step ran rows for, counting each conversation once
per step. `conversations / steps` is the average number writing together.
Paused or cancelled conversations and those with no room for a token do not count.
Steps that fail before finishing are excluded from these totals. `routings` counts each
row's trip to each routed expert (the shared expert is not counted), and
`experts` the experts those trips reached, each counted once per layer and
step, so rows that share an expert count it once. Both are counted only in steps
whose routings the model counts on the host (not while every expert is in VRAM);
`routed_rows` is the rows of those steps. `cpu_experts` is the experts the CPU
computed in the steps, counted the same way. `guesses` is the guessed tokens
checked (from the drafter and suffix lookup) and `kept` those the model kept.
`guess_room` sums the guess row allowance for the conversation-steps counted in
`conversations`: each one's share of spare rows, capped by the lookahead and its
remaining output tokens after its own next token, or zero when drafting is off. This is an upper bound
before draft source limits, confidence cuts, cooling and memory constraints.
Dividing by `conversations` gives the average allowance per conversation per step,
not a maximum for every conversation or the current settings.
`prefill.tokens` is the prompt
tokens read and `prefill.ms` the time spent reading them, without the turns the
prompt gives to conversations that decode meanwhile. Readers ignore keys they do not know, so that more statistics can be
added.

Python retains its old protocol client for installed engines that do not
advertise `requests`. It never chooses that client for an updated engine.

Build `request_protocol_driver` and `request_test` to exercise the shared C++
transport without loading a model.
Run `python -m pytest strata_server/test_request_protocol.py`. Set `STRATA_REQUEST_PROTOCOL_DRIVER` to
the executable's full path when the build is outside `build` or `build-hip-win`.
These checks cover event routing, settings during generation, lazy startup,
cancellation, preemption continuation, admission deadlines and late replies.
They do not establish model-backed GPU correctness.
