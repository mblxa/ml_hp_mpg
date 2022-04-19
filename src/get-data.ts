import fetch from 'node-fetch';
import * as tf from "@tensorflow/tfjs-node";
import {jsonData} from "./csvjson";

//Id,Surname,Score,Nationality,Gender,Age,Tenure,Balance,Products,Card,Active,Salary,Exited
export type ApiUser = {
    Id: number;
    Surname: string;
    Score: number;
    Gender: 'Male' | "Female";
    Age: number;
    Tenure: number;
    Balance: number;
    Products: number;
    Card: number;
    Active: number;
    Salary: number;
    Exited: number;
}

export type TrainData = Pick<ApiUser, "Score" | "Exited" | "Tenure" | "Balance" | "Products" | "Salary">

export const getData = async (): Promise<TrainData[]> => {
    const data: ApiUser[] = jsonData as unknown as ApiUser[];
    const cleaned = data.map<TrainData>(item => ({
        Score: item.Score,
        Exited: item.Exited,
        Tenure: item.Tenure,
        Balance: item.Balance,
        Products: item.Products,
        Salary: item.Salary,
    }))

    tf.util.shuffle(cleaned)
    return cleaned;
}
