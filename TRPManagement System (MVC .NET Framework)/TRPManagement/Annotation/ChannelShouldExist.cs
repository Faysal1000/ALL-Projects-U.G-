using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;
using TRPManagement.Entity_Framework;

namespace TRPManagement.Annotation
{
    [AttributeUsage(AttributeTargets.Property |
    AttributeTargets.Field, AllowMultiple = true)]
    public class ChannelShouldExist : ValidationAttribute
    {
        protected override ValidationResult IsValid(object value, ValidationContext validationContext)
        {
            var _channelId = (int)value;

            // Accessing database and cehking
            TRP_ManagementEntities db = new TRP_ManagementEntities();

            var exists = (from data in db.Channels
                          where data.ChannelId == _channelId
                          select data).SingleOrDefault();
            if (exists == null)
            {
                return new ValidationResult("Channel Name Doesn't Exist");
            }
            return ValidationResult.Success;

        }
    }
}