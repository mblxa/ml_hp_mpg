import {TrainData} from "./get-data";
import {oneHot} from "./one-hot";

export const toCategorical = (data: TrainData[], column: keyof TrainData): number[][] => {
    const values = data.map((r) => r[column as keyof TrainData]);
    const uniqueValues = new Set(values);

    const mapping: Record<string, number> = {};

    Array.from(uniqueValues).forEach((i, v) => {
        mapping[i] = v;
    });

    const encoded = values
        .map((v) => {
            if (!v) {
                return 0;
            }
            return mapping[v];
        })
        .map((v) => oneHot(v, uniqueValues.size));
    return encoded;
};
