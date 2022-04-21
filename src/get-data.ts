import {jsonData} from "./csvjson";

//Id,Surname,Score,Nationality,Gender,Age,Tenure,Balance,Products,Card,Active,Salary,Exited
export type ApiUser = {
    Id: number;
    Score: number;
    Tenure: number;
    Balance: number;
    Salary: number;
    Surname: string;
    Gender: 'Male' | "Female";
    Age: number;
    Products: number;
    Card: number;
    Active: number;
    Exited: number;
}

export type TrainData = ApiUser;

export const getData = async (): Promise<TrainData[]> => {
    return (jsonData as unknown as ApiUser[]).filter(item => item.Salary > 0 && item.Balance > 0);
}
