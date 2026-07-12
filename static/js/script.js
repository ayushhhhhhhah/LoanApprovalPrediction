const form = document.getElementById("loanForm");

form.addEventListener("submit", async function (event) {

    event.preventDefault();

    const data = {

        Gender: Number(document.getElementById("Gender").value),
        Married: Number(document.getElementById("Married").value),
        Dependents: Number(document.getElementById("Dependents").value),
        Education: Number(document.getElementById("Education").value),
        Self_Employed: Number(document.getElementById("Self_Employed").value),
        ApplicantIncome: Number(document.getElementById("ApplicantIncome").value),
        CoapplicantIncome: Number(document.getElementById("CoapplicantIncome").value),
        LoanAmount: Number(document.getElementById("LoanAmount").value),
        Loan_Amount_Term: Number(document.getElementById("Loan_Amount_Term").value),
        Credit_History: Number(document.getElementById("Credit_History").value),
        Property_Area: Number(document.getElementById("Property_Area").value)
    };

    const response = await fetch("/predict", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify(data)

    });

    const result = await response.json();

    const resultDiv = document.getElementById("result");

    resultDiv.style.display = "block";

    if (result.prediction === "Loan Approved") {

        resultDiv.className = "success";

    } else {

        resultDiv.className = "error";

    }

    resultDiv.innerHTML = `
        <h2>${result.prediction}</h2>
        <p>Confidence : ${result.confidence}%</p>
    `;

});