export type Model={name:string;color:string;rows:number;status:string;error?:string|null;first_date?:string|null;last_date?:string|null;path?:string};
export type Metric={mae:number|null;rmse:number|null;mape:number|null;smape:number|null;mase:number|null;r2:number|null;directional_accuracy:number|null;bias:number|null;max_error:number|null;median_abs_error:number|null;p90_abs_error:number|null;rows:number};
export type Point={Date:string;Actual:number;[key:string]:number|string|null};
