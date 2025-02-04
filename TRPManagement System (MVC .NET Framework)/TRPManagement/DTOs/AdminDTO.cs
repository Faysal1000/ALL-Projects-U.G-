using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace TRPManagement.DTOs
{
    public class AdminDTO
    {
        [Required]
        public String username { get; set; }
        [Required]
        public String password { get; set; }

    }
}