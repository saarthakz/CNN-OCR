export default function (sumArr: number[]) {

  const divIndices = new Array<number>();
  let start = 0;
  for (let idx = 0; idx < sumArr.length; idx++) {
    if (sumArr[idx] != 0 && start != -1) {
      const mid = Math.floor((start + idx - 1) / 2);
      divIndices.push(mid);
      start = -1;
      continue;
    };
    if (sumArr[idx] == 0 && start == -1) {
      start = idx;
    };
  };

  if (start != -1) {
    const mid = Math.floor((start + sumArr.length - 1) / 2);
    divIndices.push(mid);
  };

  return divIndices;
};