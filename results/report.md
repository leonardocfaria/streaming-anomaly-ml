20000 events, level shift (+60 % latency) at event 10000, 0.5 % injected anomalies, seed 0.

| run | segment | precision | recall | false alarms | events |
|---|---|---|---|---|---|
| drift handling ON | before drift | 1.000 | 1.000 | 0 | 10000 |
| drift handling ON | first after drift | 0.286 | 0.667 | 5 | 500 |
| drift handling ON | later after drift | 1.000 | 1.000 | 0 | 9500 |
| drift handling OFF | before drift | 1.000 | 1.000 | 0 | 10000 |
| drift handling OFF | first after drift | 0.006 | 1.000 | 497 | 500 |
| drift handling OFF | later after drift | 0.005 | 1.000 | 9453 | 9500 |

drift handling ON: drift detections at events [10006]; last false alarm 4 events after the shift; 19697 per-event model updates; 129,057 events/s through the threaded queue pipeline.

drift handling OFF: drift detections at events []; last false alarm 9999 events after the shift; 9752 per-event model updates; 183,412 events/s through the threaded queue pipeline.
