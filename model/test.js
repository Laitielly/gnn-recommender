import http from 'k6/http';
import { sleep, check } from 'k6';


export let options = {
    vus: 5000,
    duration: '20m',
};

export default function () {
    const url = 'http://localhost:8080/recommendations';

    const payload = JSON.stringify({
        user_id: Math.floor(Math.random() * 10000)
    });

    const params = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    const res = http.post(url, payload, params);

    check(res, {
        'Status is 200': (r) => r.status === 200,
        'Response time is < 200ms': (r) => r.timings.duration < 200,
    });

    sleep(1);
}
