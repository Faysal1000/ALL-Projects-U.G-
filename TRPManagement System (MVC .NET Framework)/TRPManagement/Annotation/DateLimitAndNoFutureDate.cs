using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace TRPManagement.Annotation
{
    [AttributeUsage(AttributeTargets.Property |
AttributeTargets.Field, AllowMultiple = true)]
    public class DateLimitAndNoFutureDate:ValidationAttribute
    {
        protected override ValidationResult IsValid(object value, ValidationContext validationContext)
        {
            var year = (int)value; 

            if (year>=1900 && year <= DateTime.Now.Year)
            {
                return ValidationResult.Success;
            }
            else
            {
                return new ValidationResult("The date must be in between 1900 and "+DateTime.Now.Year.ToString());
            }
        }
    }
}