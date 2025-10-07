using StockDamageTask.Context;
using StockDamageTask.DTOs;
using StockDamageTask.Models;
using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Linq;
using System.Web.Mvc;

namespace StockDamageTask.Controllers
{
    public class StockDamageController : Controller
    {
        // database helper object so no need to open and close connection manually
        DatabaseHelper databaseHelper = new DatabaseHelper();

        // generic class to convert datatable each row to its corresponding class/object
        public List<T> ConvertToList<T>(DataTable dt) where T : new()
        {
            List<T> list = new List<T>();

            foreach (DataRow row in dt.Rows)
            {
                T obj = new T();
                foreach (var prop in typeof(T).GetProperties())
                {
                    if (!dt.Columns.Contains(prop.Name) || row[prop.Name] == DBNull.Value)
                        continue;

                    Type targetType = Nullable.GetUnderlyingType(prop.PropertyType) ?? prop.PropertyType;

                    object safeValue = Convert.ChangeType(row[prop.Name], targetType);

                    prop.SetValue(obj, safeValue);
                }
                list.Add(obj);
            }
            return list;
        }

        // landing page controller.
        [HttpGet]
        public ActionResult StockDamageMaintain()
        {
            // retriving all the data from database
            string getAllCurrencyQuery = "Select * from currency";
            string getAllEmployeeQuery = "Select * from employee";
            string getAllGodownQuery = "Select * from godown";
            string getAllStockQuery = "Select * from stock";
            string getStockDamageQuery = "Select * from StockDamage";
            string getSubItemCodeQuery = "Select * from subitemcode";

            // here converting the data table to its corresponding class and getting list
            var allCurrency= ConvertToList<Currency>(databaseHelper.ExecuteQuery(getAllCurrencyQuery));
            var allEmployees = ConvertToList<Employee>(databaseHelper.ExecuteQuery(getAllEmployeeQuery));
            var allGodowns = ConvertToList<Godown>(databaseHelper.ExecuteQuery(getAllGodownQuery));
            var allStocks = ConvertToList<Stock>(databaseHelper.ExecuteQuery(getAllStockQuery));
            var allStockDamages = ConvertToList<StockDamage>(databaseHelper.ExecuteQuery(getStockDamageQuery));
            var allSubItemCodes = ConvertToList<SubItemCodes>(databaseHelper.ExecuteQuery(getSubItemCodeQuery));

            // for strok damage, sending only necessary details 
            // it uses a DTO and convert the data in dto list for client side use
            var allStockDamagesDTOConverted = new List<StockDamageDTO>();
            foreach (var sd in allStockDamages)
            {
                allStockDamagesDTOConverted.Add(new StockDamageDTO()
                {
                    VoucherNo = sd.VoucherNo,
                    // here warehousename is coming from godown id mapping
                    WarehouseName = (from g in allGodowns
                                     where g.GodownNo == sd.GodownNo
                                     select g.GodownName).FirstOrDefault(),
                    ItemName = sd.ItemName,
                    ItemCode = sd.ItemCode,
                    BatchNo = sd.BatchNo,
                    Currency = sd.CurrencyName,
                    Quantity = sd.Quantity,
                    Rate = sd.Rate,
                    Amount_in = sd.Amount_in,
                    ExchangeRate = sd.ConversionRate,
                    TotalAmount = sd.TotalAmount
                });
            }

            // all the list is added to viewbag for accesing in the client side
            ViewBag.Currency = allCurrency;
            ViewBag.Employees = allEmployees;
            ViewBag.Godowns = allGodowns;
            ViewBag.Stocks = allStocks;
            ViewBag.StockDamages = allStockDamagesDTOConverted;
            ViewBag.SubItemCodes = allSubItemCodes;

            return View(); // returing view landing page
        }

        // this method will receive data from client side and create/update/delete data via procedure
        [HttpPost]
        public ActionResult SaveStockDamage(SaveStockDamageRequest request)
        {
            try
            {
                // if somehow no content came then it will return error
                if (request == null)
                {
                    return Json(new { Status = "Error", Message = "Invalid request. Model binding failed." });
                }

                var results = new List<object>();

                if (request.addedOrEditedStockDamages != null && request.addedOrEditedStockDamages.Count > 0)
                {
                    foreach (var item in request.addedOrEditedStockDamages)
                    {
                        // making sql parameter to pass in the procedure
                        // also making sure again no invalid data came
                        SqlParameter[] addParams = new SqlParameter[]
                        {
                            new SqlParameter("@VoucherNo", item.VoucherNo == 0 ? (object)DBNull.Value : item.VoucherNo),
                            new SqlParameter("@GodownNo", string.IsNullOrEmpty(item.GodownNo) ? (object)DBNull.Value : item.GodownNo),
                            new SqlParameter("@ItemName", string.IsNullOrEmpty(item.ItemName) ? (object)DBNull.Value : item.ItemName),
                            new SqlParameter("@ItemCode", string.IsNullOrEmpty(item.ItemCode) ? (object)DBNull.Value : item.ItemCode),
                            new SqlParameter("@BatchNo", item.BatchNo == 0 ? (object)DBNull.Value : item.BatchNo),
                            new SqlParameter("@CurrencyName", string.IsNullOrEmpty(item.CurrencyName) ? (object)DBNull.Value : item.CurrencyName),
                            new SqlParameter("@ConversionRate", item.ConversionRate == 0 ? (object)DBNull.Value : item.ConversionRate),
                            new SqlParameter("@StockQuantity", item.StockQuantity == 0 ? (object)DBNull.Value : item.StockQuantity),
                            new SqlParameter("@Quantity", item.Quantity == 0 ? (object)DBNull.Value : item.Quantity),
                            new SqlParameter("@Rate", item.Rate == 0 ? (object)DBNull.Value : item.Rate),
                            new SqlParameter("@Amount_in", item.Amount_in == 0 ? (object)DBNull.Value : item.Amount_in),
                            new SqlParameter("@DrACHead", string.IsNullOrEmpty(item.DrACHead) ? (object)"Stock Damage" : item.DrACHead),
                            new SqlParameter("@EmployeeId", item.EmployeeId == 0 ? (object)DBNull.Value : item.EmployeeId),
                            new SqlParameter("@CreatedDate", item.CreatedDate == DateTime.MinValue ? (object)DateTime.Now : item.CreatedDate),
                            new SqlParameter("@Comments", string.IsNullOrEmpty(item.Comments) ? (object)DBNull.Value : item.Comments),
                            new SqlParameter("@TotalAmount", item.TotalAmount == 0 ? (object)DBNull.Value : item.TotalAmount),
                            new SqlParameter("@DeleteVoucher", 0)
                        };

                        // calling procedure througn database helper
                        // it will return every action result which is success or failed with reason
                        var dt = databaseHelper.ExecuteQueryWithParameters("SP_StockDamage_Save", addParams, CommandType.StoredProcedure);

                        if (dt != null && dt.Rows.Count > 0)
                        {   
                            // making table to send in client side- currently it can only be used by developer
                            // not for client
                            var row = dt.Rows[0];
                            results.Add(new
                            {
                                VoucherNo = item.VoucherNo,
                                Status = dt.Columns.Contains("Status") ? row["Status"].ToString() : null,
                                Message = dt.Columns.Contains("Message") ? row["Message"].ToString() : null,
                                NewVoucher = dt.Columns.Contains("VoucherNo") ? row["VoucherNo"] : null
                            });
                        }
                        else
                        {
                            // if adding data failed then error sms send
                            results.Add(new { VoucherNo = item.VoucherNo, Status = "Error", Message = "Stored procedure returned no data" });
                        }
                    }
                }

                // here the deletion is handled using voucherNo as it is primary key
                if (request.deletedVoucherNos != null && request.deletedVoucherNos.Count > 0)
                {
                    foreach (var v in request.deletedVoucherNos)
                    {
                        int voucherNo;
                        // making sure every voucher is int
                        // if string came then it will give error
                        if (int.TryParse(v, out voucherNo))
                        {
                            SqlParameter[] deleteParams = new SqlParameter[]
                            {
                                new SqlParameter("@VoucherNo", voucherNo),
                                new SqlParameter("@DeleteVoucher", 1)
                            };

                            var dt = databaseHelper.ExecuteQueryWithParameters("SP_StockDamage_Save", deleteParams, CommandType.StoredProcedure);

                            if (dt != null && dt.Rows.Count > 0)
                            {
                                var row = dt.Rows[0];
                                results.Add(new
                                {
                                    VoucherNo = voucherNo,
                                    Status = dt.Columns.Contains("Status") ? row["Status"].ToString() : null,
                                    Message = dt.Columns.Contains("Message") ? row["Message"].ToString() : null
                                });
                            }
                            else
                            {
                                results.Add(new { VoucherNo = voucherNo, Status = "Error", Message = "Stored procedure returned nothing for delete" });
                            }
                        }
                    }
                }

                // if everything is success then success message will be sent
                return Json(new { Status = "Success", Message = "Processed successfully.", Details = results });
            }
            catch (Exception ex)
            {
                // server side error will be send from here
                return Json(new { Status = "Error", Message = ex.Message });
            }
        }
    }


    // This code is done by Faysal Ahmmed - mail:faysalahmmed4200 @gmail.com
}